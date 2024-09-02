from fusemap.model import Fuse_network
from fusemap.preprocess import *
from fusemap.dataset import *
from fusemap.loss import *
from fusemap.config import *
from fusemap.utils import *
from fusemap.train_model import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributions as D
from pathlib import Path
import itertools
import dgl.dataloading as dgl_dataload
import random
import os
import anndata as ad
import torch
import numpy as np
from tqdm import tqdm
import scanpy as sc
import dgl

try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle


def spatial_integrate(
    X_input,
    save_dir,
    kneighbor,
    input_identity,
    data_pth=None,
):
    ### preprocess
    if "spatial_input" in X_input[0].obsm:
        preprocess_save = True
    else:
        preprocess_save = False
    ModelType.preprocess_save = preprocess_save
    ModelType.data_pth = data_pth
    ModelType.save_dir = save_dir
    ModelType.kneighbor = kneighbor
    ModelType.input_identity = input_identity
    ModelType.snapshot_path = f"{ModelType.save_dir}/snapshot.pt"
    Path(f"{ModelType.save_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{ModelType.save_dir}/trained_model").mkdir(parents=True, exist_ok=True)

    ModelType.n_atlas = len(X_input)
    if ModelType.preprocess_save == False:
        preprocess_raw(
            X_input,
            ModelType.kneighbor,
            ModelType.input_identity,
            ModelType.use_input.value,
            ModelType.n_atlas,
            ModelType.data_pth,
        )
    for i in range(ModelType.n_atlas):
        X_input[i].var.index = [i.upper() for i in X_input[i].var.index]
    adatas = X_input

    ModelType.n_obs = [adatas[i].shape[0] for i in range(ModelType.n_atlas)]
    ModelType.input_dim = [adatas[i].n_vars for i in range(ModelType.n_atlas)]
    ModelType.var_name = [list(i.var.index) for i in adatas]

    all_unique_genes = sorted(list(get_allunique_gene_names(*ModelType.var_name)))
    print(
        f"number of genes in each section:{[len(i) for i in ModelType.var_name]}, Number of all genes: {len(all_unique_genes)}"
    )

    ### create model
    model = Fuse_network(
        ModelType.pca_dim.value,
        ModelType.input_dim,
        ModelType.hidden_dim.value,
        ModelType.latent_dim.value,
        ModelType.dropout_rate.value,
        ModelType.var_name,
        all_unique_genes,
        ModelType.use_input.value,
        ModelType.harmonized_gene,
        ModelType.n_atlas,
        ModelType.input_identity,
        ModelType.n_obs,
        ModelType.n_epochs.value,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ModelType.epochs_run_pretrain = 0
    ModelType.epochs_run_final = 0
    if os.path.exists(ModelType.snapshot_path):
        print("Loading snapshot")
        load_snapshot(model, ModelType.snapshot_path, device)

    ### construct graph and data
    adj_all, g_all = construct_data(
        ModelType.n_atlas, adatas, ModelType.input_identity, model
    )
    feature_all = [
        get_feature_sparse(device, adata.obsm["spatial_input"]) for adata in adatas
    ]
    spatial_dataset_list = [
        CustomGraphDataset(i, j, ModelType.use_input) for i, j in zip(g_all, adatas)
    ]
    spatial_dataloader = CustomGraphDataLoader(
        spatial_dataset_list,
        dgl_dataload.MultiLayerFullNeighborSampler(1),
        ModelType.batch_size.value,
        shuffle=True,
        n_atlas=ModelType.n_atlas,
        drop_last=False,
    )
    spatial_dataloader_test = CustomGraphDataLoader(
        spatial_dataset_list,
        dgl_dataload.MultiLayerFullNeighborSampler(1),
        ModelType.batch_size.value,
        shuffle=False,
        n_atlas=ModelType.n_atlas,
        drop_last=False,
    )
    train_mask, val_mask = construct_mask(
        ModelType.n_atlas, spatial_dataset_list, g_all
    )

    ### train
    flagconfig = FlagConfig()
    if os.path.exists(f"{ModelType.save_dir}/lambda_disc_single.pkl"):
        with open(f"{ModelType.save_dir}/lambda_disc_single.pkl", "rb") as openfile:
            flagconfig.lambda_disc_single = pickle.load(openfile)

    if not os.path.exists(
        f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt"
    ):
        print(
            "---------------------------------- Phase 1. Pretrain FuseMap model ----------------------------------"
        )
        pretrain_model(
            model,
            spatial_dataloader,
            feature_all,
            adj_all,
            device,
            train_mask,
            val_mask,
            flagconfig,
        )

    if not os.path.exists(
        f"{ModelType.save_dir}/latent_embeddings_all_single_pretrain.pkl"
    ):
        print(
            "---------------------------------- Phase 2. Evaluate pretrained FuseMap model ----------------------------------"
        )
        if os.path.exists(
            f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt"
        ):
            read_model(
                model,
                spatial_dataloader_test,
                g_all,
                feature_all,
                adj_all,
                device,
                ModelType,
                mode="pretrain",
            )
        else:
            raise ValueError("No pretrained model!")

    if not os.path.exists(f"{ModelType.save_dir}/balance_weight_single.pkl"):
        print(
            "---------------------------------- Phase 3. Estimate_balancing_weight ----------------------------------"
        )
        balance_weight(model, adatas, ModelType.save_dir, ModelType.n_atlas, device)

    if not os.path.exists(
        f"{ModelType.save_dir}/trained_model/FuseMap_final_model_final.pt"
    ):
        model.load_state_dict(
            torch.load(
                f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt"
            )
        )
        print(
            "---------------------------------- Phase 4. Train final FuseMap model ----------------------------------"
        )
        train_model(
            model,
            spatial_dataloader,
            feature_all,
            adj_all,
            device,
            train_mask,
            val_mask,
            flagconfig,
        )

    if not os.path.exists(
        f"{ModelType.save_dir}/latent_embeddings_all_single_final.pkl"
    ):
        print(
            "---------------------------------- Phase 5. Evaluate final FuseMap model ----------------------------------"
        )
        if os.path.exists(
            f"{ModelType.save_dir}/trained_model/FuseMap_final_model_final.pt"
        ):
            read_model(
                model,
                spatial_dataloader_test,
                g_all,
                feature_all,
                adj_all,
                device,
                ModelType,
                mode="final",
            )
        else:
            raise ValueError("No final model!")

    print(
        "---------------------------------- Finish ----------------------------------"
    )

    ### read out gene embedding
    read_gene_embedding(
        model,
        all_unique_genes,
        ModelType.save_dir,
        ModelType.n_atlas,
        ModelType.var_name,
    )

    ### read out cell embedding
    annotation_transfer(
        adatas,
        ModelType.save_dir,
    )

    return
