import logging
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
import os
import anndata as ad
import torch
import numpy as np
from tqdm import tqdm
import scanpy as sc

try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle


def spatial_map(
    molccf_path,
    X_input,
    args,
    kneighbor,
    input_identity,
    data_pth=None,
):
    ### preprocess
    ModelType.data_pth = data_pth
    ModelType.save_dir = args.output_save_dir
    ModelType.kneighbor = kneighbor
    ModelType.input_identity = input_identity
    ModelType.snapshot_path = f"{ModelType.save_dir}/snapshot.pt"
    Path(f"{ModelType.save_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{ModelType.save_dir}/trained_model").mkdir(parents=True, exist_ok=True)

    ModelType.n_atlas = len(X_input)
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
    logging.info(
        f"\n\nnumber of genes in each section:{[len(i) for i in ModelType.var_name]}, Number of all genes: {len(all_unique_genes)}\n"
    )

    ### load pretrain model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        TRAINED_MODEL,
        TRAINED_X_NUM,
        TRAINED_GENE_EMBED,
        TRAINED_GENE_NAME,
    ) = load_ref_model(molccf_path, device)

    ### define new model
    PRETRAINED_GENE = []
    new_train_gene = []
    for i in all_unique_genes:
        if i not in TRAINED_GENE_NAME:
            new_train_gene.append(i)
        else:
            PRETRAINED_GENE.append(i)
    pretrain_index = [TRAINED_GENE_NAME.index(i) for i in PRETRAINED_GENE]
    logging.info(
        f"\n\npretrain gene number:{len(PRETRAINED_GENE)}, new gene number:{len(new_train_gene)}\n"
    )

    adapt_model = Fuse_network(
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
        pretrain_model=True,
        pretrain_n_atlas=TRAINED_X_NUM,
        PRETRAINED_GENE=PRETRAINED_GENE,
        new_train_gene=new_train_gene,
    )
    adapt_model.to(device)

    ModelType.epochs_run_pretrain = 0
    ModelType.epochs_run_final = 0
    if os.path.exists(ModelType.snapshot_path):
        logging.info("\n\nLoading snapshot\n")
        load_snapshot(adapt_model, ModelType.snapshot_path, device)

    ### construct graph and data
    adj_all, g_all = construct_data(
        ModelType.n_atlas, adatas, ModelType.input_identity, adapt_model
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
        f"{ModelType.save_dir}/trained_model/FuseMap_map_model_final.pt"
    ):
        logging.info(
            "\n\n---------------------------------- Phase 1. Map FuseMap model ----------------------------------\n"
        )
        ### transfer model weight
        transfer_weight(TRAINED_MODEL, pretrain_index, adapt_model)

        ### load reference data
        (dataloader_pretrain_single, dataloader_pretrain_spatial) = load_ref_data(
            molccf_path,
            TRAINED_X_NUM,
            ModelType.batch_size.value,
            USE_REFERENCE_PCT=ModelType.USE_REFERENCE_PCT.value
        )

        map_model(
            adapt_model,
            spatial_dataloader,
            feature_all,
            adj_all,
            device,
            train_mask,
            val_mask,
            molccf_path,
            dataloader_pretrain_single,
            dataloader_pretrain_spatial,
            TRAINED_X_NUM,
            flagconfig,
        )

    if not os.path.exists(
        f"{ModelType.save_dir}/latent_embeddings_all_single_map.pkl"
    ):
        logging.info(
            "\n\n---------------------------------- Phase 2. Evaluate mapped FuseMap model ----------------------------------\n"
        )
        if os.path.exists(
            f"{ModelType.save_dir}/trained_model/FuseMap_map_model_final.pt"
        ):
            read_model(
                adapt_model,
                spatial_dataloader_test,
                g_all,
                feature_all,
                adj_all,
                device,
                ModelType,
                mode="map",
            )
        else:
            raise ValueError("No mapped model!")


    logging.info(
        "\n\n---------------------------------- Finish ----------------------------------\n"
    )

    ### read out gene embedding
    read_gene_embedding_map(
        adapt_model,
        new_train_gene,
        PRETRAINED_GENE,
        ModelType.save_dir,
        ModelType.n_atlas,
        ModelType.var_name,
    )

    ### read out cell embedding
    read_cell_embedding(
        adatas,
        ModelType.save_dir,
        args.keep_celltype,
        args.keep_tissueregion,
        use_key='map',
    )

    ### transfer molCCF cell annotations
    transfer_annotation(
        adatas,
        ModelType.save_dir,
        molccf_path,
    )
    return
