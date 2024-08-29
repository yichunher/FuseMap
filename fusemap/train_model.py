from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributions as D
from pathlib import Path
import itertools
import dgl.dataloading as dgl_dataload
import random
import os
from fusemap.config import *
from fusemap.utils import *
from fusemap.loss import *
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
    
    
def get_data(blocks_all, feature_all, adj_all, train_mask, device, model):
    row_index_all = {}
    col_index_all = {}
    for i_atlas in range(ModelType.n_atlas):
        row_index = list(blocks_all[i_atlas]["spatial"][0])
        col_index = list(blocks_all[i_atlas]["spatial"][1])
        row_index_all[i_atlas] = torch.sort(torch.vstack(row_index).flatten())[
            0
        ].tolist()
        col_index_all[i_atlas] = torch.sort(torch.vstack(col_index).flatten())[
            0
        ].tolist()

    batch_features_all = [
        torch.FloatTensor(feature_all[i][row_index_all[i], :].toarray()).to(
            device
        )
        for i in range(ModelType.n_atlas)
    ]

    adj_all_block = [
        torch.FloatTensor(
            adj_all[i][row_index_all[i], :]
            .tocsc()[:, col_index_all[i]]
            .todense()
        ).to(device)
        if ModelType.input_identity[i] == "ST"
        else model.scrna_seq_adj["atlas" + str(i)]()[row_index_all[i], :][
            :, col_index_all[i]
        ]
        for i in range(ModelType.n_atlas)
    ]
    adj_all_block_dis = [
        torch.FloatTensor(
            adj_all[i][row_index_all[i], :]
            .tocsc()[:, col_index_all[i]]
            .todense()
        ).to(device)
        if ModelType.input_identity[i] == "ST"
        else model.scrna_seq_adj["atlas" + str(i)]()[row_index_all[i], :][
            :, col_index_all[i]
        ].detach()
        for i in range(ModelType.n_atlas)
    ]

    train_mask_batch_single = [
        train_mask_i[row_index_all[blocks_all_ind]]
        for train_mask_i, blocks_all_ind in zip(train_mask, blocks_all)
    ]
    train_mask_batch_spatial = [
        train_mask_i[col_index_all[blocks_all_ind]]
        for train_mask_i, blocks_all_ind in zip(train_mask, blocks_all)
    ]

    ### discriminator flags
    flag_shape_single = [len(row_index_all[i]) for i in range(ModelType.n_atlas)]
    flag_all_single = torch.cat(
        [torch.full((x,), i) for i, x in enumerate(flag_shape_single)]
    )
    flag_source_cat_single = flag_all_single.long().to(device)

    flag_shape_spatial = [len(col_index_all[i]) for i in range(ModelType.n_atlas)]
    flag_all_spatial = torch.cat(
        [torch.full((x,), i) for i, x in enumerate(flag_shape_spatial)]
    )
    flag_source_cat_spatial = flag_all_spatial.long().to(device)
    return (batch_features_all, adj_all_block, adj_all_block_dis, 
            train_mask_batch_single, train_mask_batch_spatial, 
            flag_source_cat_single, flag_source_cat_spatial,
            row_index_all, col_index_all)

def pretrain_model(
    model, spatial_dataloader, feature_all, adj_all, device, train_mask, val_mask, flagconfig
):
    loss_atlas_val_best = float("inf")
    patience_counter = 0  # 2

    optimizer_dis = getattr(torch.optim, ModelType.optim_kw.value)(
        itertools.chain(
            model.discriminator_single.parameters(),
            model.discriminator_spatial.parameters(),
        ),
        lr=ModelType.learning_rate.value,
    )
    optimizer_ae = getattr(torch.optim, ModelType.optim_kw.value)(
        itertools.chain(
            model.encoder.parameters(),
            model.decoder.parameters(),
            model.scrna_seq_adj.parameters(),
        ),
        lr=ModelType.learning_rate.value,
    )
    scheduler_dis = ReduceLROnPlateau(
        optimizer_dis,
        mode="min",
        factor=ModelType.lr_factor_pretrain.value,
        patience=ModelType.lr_patience_pretrain.value,
        verbose=True,
    )
    scheduler_ae = ReduceLROnPlateau(
        optimizer_ae,
        mode="min",
        factor=ModelType.lr_factor_pretrain.value,
        patience=ModelType.lr_patience_pretrain.value,
        verbose=True,
    )
    

    for epoch in tqdm(range(ModelType.epochs_run_pretrain + 1, ModelType.n_epochs.value)):
        loss_dis = 0
        loss_ae_dis = 0
        loss_all_item = 0
        loss_atlas_i = {}
        for i in range(ModelType.n_atlas):
            loss_atlas_i[i] = 0
        loss_atlas_val = 0
        anneal = max(1 - (epoch - 1) / flagconfig.align_anneal, 0) if flagconfig.align_anneal else 0

        model.train()

        # random_numbers = random.sample(range(len(spatial_dataloader)+1), 1000)

        for ind,blocks_all in enumerate(spatial_dataloader):
            # if ind not in random_numbers:
            #     continue
            
            (batch_features_all, adj_all_block, adj_all_block_dis, 
            train_mask_batch_single, train_mask_batch_spatial, 
            flag_source_cat_single, flag_source_cat_spatial,_,_) = get_data(blocks_all, feature_all, adj_all, 
                                                                        train_mask, device, model)

            # Train discriminator part
            loss_part1 = compute_dis_loss_pretrain(
                model,
                flag_source_cat_single,
                flag_source_cat_spatial,
                anneal,
                batch_features_all,
                adj_all_block_dis,
                train_mask_batch_single,
                train_mask_batch_spatial,
                flagconfig,
            )
            model.zero_grad(set_to_none=True)
            loss_part1["dis"].backward()
            optimizer_dis.step()
            loss_dis += loss_part1["dis"].item()

            # Train AE part
            loss_part2 = compute_ae_loss_pretrain(
                model,
                flag_source_cat_single,
                flag_source_cat_spatial,
                anneal,
                batch_features_all,
                adj_all_block,
                train_mask_batch_single,
                train_mask_batch_spatial,
                flagconfig,
            )
            model.zero_grad(set_to_none=True)
            loss_part2["loss_all"].backward()
            optimizer_ae.step()

            for i in range(ModelType.n_atlas):
                loss_atlas_i[i] += loss_part2["loss_AE_all"][i].item()
            loss_all_item += loss_part2["loss_all"].item()
            loss_ae_dis += loss_part2["dis_ae"].item()

        flagconfig.align_anneal /= 2

        if ModelType.verbose == True:
            print(
                f"Train Epoch {epoch}/{ModelType.n_epochs}, \
            Loss dis: {loss_dis / len(spatial_dataloader)},\
            Loss AE: {[i / len(spatial_dataloader) for i in loss_atlas_i.values()]} , \
            Loss ae dis:{loss_ae_dis / len(spatial_dataloader)},\
            Loss all:{loss_all_item / len(spatial_dataloader)}"
            )

        save_snapshot(model, epoch, ModelType.epochs_run_final, ModelType.snapshot_path)

        if not os.path.exists(f"{ModelType.save_dir}/lambda_disc_single.pkl"):
            save_obj(flagconfig.lambda_disc_single, f"{ModelType.save_dir}/lambda_disc_single")

        ################# validation
        if epoch > ModelType.TRAIN_WITHOUT_EVAL.value:
            model.eval()
            with torch.no_grad():
                for ind,blocks_all in enumerate(spatial_dataloader):
                    # if ind not in random_numbers:
                    #     continue

                    (batch_features_all, adj_all_block, adj_all_block_dis, 
                    val_mask_batch_single, val_mask_batch_spatial, 
                    flag_source_cat_single, flag_source_cat_spatial,_,_) = get_data(blocks_all, feature_all, adj_all, 
                                                                                val_mask, device, model)
                    
                    # val AE part
                    loss_part2 = compute_ae_loss_pretrain(
                        model,
                        flag_source_cat_single,
                        flag_source_cat_spatial,
                        anneal,
                        batch_features_all,
                        adj_all_block,
                        val_mask_batch_single,
                        val_mask_batch_spatial,
                        flagconfig,
                    )

                    for i in range(ModelType.n_atlas):
                        loss_atlas_val += loss_part2["loss_AE_all"][i].item()
                        # if np.isnan(loss_part2['loss_AE_all'][i].item()):
                        #     p=0

                loss_atlas_val = loss_atlas_val / len(spatial_dataloader) / ModelType.n_atlas

                if ModelType.verbose == True:
                    print(
                        f"Validation Epoch {epoch + 1}/{ModelType.n_epochs}, \
                    Loss AE validation: {loss_atlas_val} "
                    )

            scheduler_dis.step(loss_atlas_val)
            scheduler_ae.step(loss_atlas_val)
            current_lr = optimizer_dis.param_groups[0]["lr"]
            print(f"current lr:{current_lr}")

            # If the loss is lower than the best loss so far, save the model And reset the patience counter
            if loss_atlas_val < loss_atlas_val_best:
                loss_atlas_val_best = loss_atlas_val
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model.pt",
                )

            else:
                patience_counter += 1

            # If the patience counter is greater than or equal to the patience limit, stop training
            if patience_counter >= ModelType.patience_limit_pretrain.value:
                print("Early stopping due to loss not improving - patience count")
                os.rename(
                    f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model.pt",
                    f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt",
                )
                print("File name changed")
                break
            if current_lr < ModelType.lr_limit_pretrain.value:
                print("Early stopping due to loss not improving - learning rate")
                os.rename(
                    f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model.pt",
                    f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt",
                )
                print("File name changed")
                break

        # torch.save(model.state_dict(), f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_{epoch}.pt")

    if os.path.exists(f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model.pt"):
        os.rename(
            f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model.pt",
            f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt",
        )
        print("File name changed in the end")


def train_model(
    model, spatial_dataloader, feature_all, adj_all, device, train_mask, val_mask, flagconfig
):
    with open(f"{ModelType.save_dir}/balance_weight_single.pkl", "rb") as openfile:
        balance_weight_single = pickle.load(openfile)
    with open(f"{ModelType.save_dir}/balance_weight_spatial.pkl", "rb") as openfile:
        balance_weight_spatial = pickle.load(openfile)
    balance_weight_single = [i.to(device) for i in balance_weight_single]
    balance_weight_spatial = [i.to(device) for i in balance_weight_spatial]

    loss_atlas_val_best = float("inf")
    patience_counter = 0

    optimizer_dis = getattr(torch.optim, ModelType.optim_kw.value)(
        itertools.chain(
            model.discriminator_single.parameters(),
            model.discriminator_spatial.parameters(),
        ),
        lr=ModelType.learning_rate.value,
    )
    optimizer_ae = getattr(torch.optim, ModelType.optim_kw.value)(
        itertools.chain(
            model.encoder.parameters(),
            model.decoder.parameters(),
            model.scrna_seq_adj.parameters(),
        ),
        lr=ModelType.learning_rate.value,
    )
    scheduler_dis = ReduceLROnPlateau(
        optimizer_dis,
        mode="min",
        factor=ModelType.lr_factor_final.value,
        patience=ModelType.lr_patience_final.value,
        verbose=True,
    )
    scheduler_ae = ReduceLROnPlateau(
        optimizer_ae,
        mode="min",
        factor=ModelType.lr_factor_final.value,
        patience=ModelType.lr_patience_final.value,
        verbose=True,
    )

    for epoch in tqdm(range(ModelType.epochs_run_final + 1, ModelType.n_epochs.value)):
        loss_dis = 0
        loss_ae_dis = 0
        loss_all_item = 0
        loss_atlas_i = {}
        for i in range(ModelType.n_atlas):
            loss_atlas_i[i] = 0
        loss_atlas_val = 0
        anneal = max(1 - (epoch - 1) / flagconfig.align_anneal, 0) if flagconfig.align_anneal else 0

        model.train()
                
        # random_numbers = random.sample(range(len(spatial_dataloader)+1), 1000)
            
        for ind,blocks_all in enumerate(spatial_dataloader):
            # if ind not in random_numbers:
            #     continue

            (batch_features_all, adj_all_block, adj_all_block_dis, 
            train_mask_batch_single, train_mask_batch_spatial, 
            flag_source_cat_single, flag_source_cat_spatial,
            row_index_all, col_index_all) = get_data(blocks_all, feature_all, adj_all, 
                                                    train_mask, device, model)



            balance_weight_single_block = [
                balance_weight_single[i][row_index_all[i]] for i in range(ModelType.n_atlas)
            ]

            balance_weight_spatial_block = [
                balance_weight_spatial[i][col_index_all[i]] for i in range(ModelType.n_atlas)
            ]

            # Train discriminator part
            loss_part1 = compute_dis_loss(
                model,
                flag_source_cat_single,
                flag_source_cat_spatial,
                anneal,
                batch_features_all,
                adj_all_block_dis,
                train_mask_batch_single,
                train_mask_batch_spatial,
                balance_weight_single_block,
                balance_weight_spatial_block,
                flagconfig,
            )
            model.zero_grad(set_to_none=True)
            loss_part1["dis"].backward()
            optimizer_dis.step()
            loss_dis += loss_part1["dis"].item()

            # Train AE part
            loss_part2 = compute_ae_loss(
                model,
                flag_source_cat_single,
                flag_source_cat_spatial,
                anneal,
                batch_features_all,
                adj_all_block,
                train_mask_batch_single,
                train_mask_batch_spatial,
                balance_weight_single_block,
                balance_weight_spatial_block,
                flagconfig,
            )
            model.zero_grad(set_to_none=True)
            loss_part2["loss_all"].backward()
            optimizer_ae.step()

            for i in range(ModelType.n_atlas):
                loss_atlas_i[i] += loss_part2["loss_AE_all"][i].item()
            loss_all_item += loss_part2["loss_all"].item()
            loss_ae_dis += loss_part2["dis_ae"].item()

        flagconfig.align_anneal /= 2

        save_snapshot(model, ModelType.epochs_run_pretrain, epoch, ModelType.snapshot_path)

        if ModelType.verbose == True:
            print(
                f"Train Epoch {epoch + 1}/{ModelType.n_epochs.value}, \
            Loss dis: {loss_dis / len(spatial_dataloader)},\
            Loss AE: {[i / len(spatial_dataloader) for i in loss_atlas_i.values()]} , \
            Loss ae dis:{loss_ae_dis / len(spatial_dataloader)},\
            Loss all:{loss_all_item / len(spatial_dataloader)}"
            )

        ################# validation
        if epoch > ModelType.TRAIN_WITHOUT_EVAL.value:
            model.eval()
            with torch.no_grad():
           
                for ind,blocks_all in enumerate(spatial_dataloader):
                    # if ind not in random_numbers:
                    #     continue

                    (batch_features_all, adj_all_block, adj_all_block_dis, 
                    val_mask_batch_single, val_mask_batch_spatial, 
                    flag_source_cat_single, flag_source_cat_spatial,
                    row_index_all, col_index_all) = get_data(blocks_all, feature_all, adj_all, 
                                                            val_mask, device, model)


                    balance_weight_single_block = [
                        balance_weight_single[i][row_index_all[i]]
                        for i in range(ModelType.n_atlas)
                    ]

                    balance_weight_spatial_block = [
                        balance_weight_spatial[i][col_index_all[i]]
                        for i in range(ModelType.n_atlas)
                    ]


                    # val AE part
                    loss_part2 = compute_ae_loss(
                        model,
                        flag_source_cat_single,
                        flag_source_cat_spatial,
                        anneal,
                        batch_features_all,
                        adj_all_block,
                        val_mask_batch_single,
                        val_mask_batch_spatial,
                        balance_weight_single_block,
                        balance_weight_spatial_block,
                        flagconfig,
                    )

                    for i in range(ModelType.n_atlas):
                        loss_atlas_val += loss_part2["loss_AE_all"][i].item()

                loss_atlas_val = loss_atlas_val / len(spatial_dataloader) / ModelType.n_atlas
                if ModelType.verbose == True:
                    print(
                        f"Validation Epoch {epoch + 1}/{ModelType.n_epochs.value}, \
                    Loss AE validation: {loss_atlas_val} "
                    )

            scheduler_dis.step(loss_atlas_val)
            scheduler_ae.step(loss_atlas_val)
            current_lr = optimizer_dis.param_groups[0]["lr"]
            print(f"current lr:{current_lr}")

            # If the loss is lower than the best loss so far, save the model And reset the patience counter
            if loss_atlas_val < loss_atlas_val_best:
                loss_atlas_val_best = loss_atlas_val
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    f"{ModelType.save_dir}/trained_model/FuseMap_final_model.pt",
                )
            else:
                patience_counter += 1

            # If the patience counter is greater than or equal to the patience limit, stop training
            if patience_counter >= ModelType.patience_limit_final.value:
                # torch.save(model.state_dict(), f"{save_dir}/trained_model/FuseMap_final_model_end.pt")
                print("Early stopping due to loss not improving")
                os.rename(
                    f"{ModelType.save_dir}/trained_model/FuseMap_final_model.pt",
                    f"{ModelType.save_dir}/trained_model/FuseMap_final_model_final.pt",
                )
                print("File name changed")
                break
            if current_lr < ModelType.lr_limit_final.value:
                # torch.save(model.state_dict(), f"{save_dir}/trained_model/FuseMap_final_model_end.pt")
                print("Early stopping due to loss not improving - learning rate")
                os.rename(
                    f"{ModelType.save_dir}/trained_model/FuseMap_final_model.pt",
                    f"{ModelType.save_dir}/trained_model/FuseMap_final_model_final.pt",
                )
                print("File name changed")
                break

    if os.path.exists(f"{ModelType.save_dir}/trained_model/FuseMap_final_model.pt"):
        os.rename(
            f"{ModelType.save_dir}/trained_model/FuseMap_final_model.pt",
            f"{ModelType.save_dir}/trained_model/FuseMap_final_model_final.pt",
        )
        print("File name changed in the end")


def read_model(
    model, spatial_dataloader_test, g_all, feature_all, adj_all, device, ModelType, mode
):
    model.load_state_dict(
        torch.load(f"{ModelType.save_dir}/trained_model/FuseMap_{mode}_model_final.pt")
    )

    with torch.no_grad():
        model.eval()

        learnt_scrna_seq_adj = {}
        for i in range(ModelType.n_atlas):
            if ModelType.input_identity[i] == "scrna":
                learnt_scrna_seq_adj["atlas" + str(i)] = (
                    model.scrna_seq_adj["atlas" + str(i)]().detach().cpu().numpy()
                )

        for blocks_all in tqdm(spatial_dataloader_test):
            row_index_all = {}
            col_index_all = {}
            for i_atlas in range(ModelType.n_atlas):
                row_index = list(blocks_all[i_atlas]["spatial"][0])
                col_index = list(blocks_all[i_atlas]["spatial"][1])
                row_index_all[i_atlas] = torch.sort(torch.vstack(row_index).flatten())[
                    0
                ].tolist()
                col_index_all[i_atlas] = torch.sort(torch.vstack(col_index).flatten())[
                    0
                ].tolist()

            batch_features_all = [
                torch.FloatTensor(feature_all[i][row_index_all[i], :].toarray()).to(
                    device
                )
                for i in range(ModelType.n_atlas)
            ]
            adj_all_block_dis = [
                torch.FloatTensor(
                    adj_all[i][row_index_all[i], :]
                    .tocsc()[:, col_index_all[i]]
                    .todense()
                ).to(device)
                if ModelType.input_identity[i] == "ST"
                else model.scrna_seq_adj["atlas" + str(i)]()[row_index_all[i], :][
                    :, col_index_all[i]
                ].detach()
                for i in range(ModelType.n_atlas)
            ]

            z_all = [
                model.encoder["atlas" + str(i)](
                    batch_features_all[i], adj_all_block_dis[i]
                )
                for i in range(ModelType.n_atlas)
            ]
            z_distribution_all = [
                D.Normal(z_all[i][0], z_all[i][1]) for i in range(ModelType.n_atlas)
            ]

            z_spatial_all = [z_all[i][2] for i in range(ModelType.n_atlas)]

            for i in range(ModelType.n_atlas):
                g_all[i].nodes[row_index_all[i]].data["single_feat_hidden"] = (
                    z_distribution_all[i].loc.detach().cpu()
                )
                g_all[i].nodes[col_index_all[i]].data["spatial_feat_hidden"] = (
                    z_spatial_all[i].detach().cpu()
                )

    latent_embeddings_all_single = [
        g_all[i].ndata["single_feat_hidden"].numpy() for i in range(ModelType.n_atlas)
    ]
    latent_embeddings_all_spatial = [
        g_all[i].ndata["spatial_feat_hidden"].numpy() for i in range(ModelType.n_atlas)
    ]

    save_obj(
        latent_embeddings_all_single,
        f"{ModelType.save_dir}/latent_embeddings_all_single_{mode}",
    )
    save_obj(
        latent_embeddings_all_spatial,
        f"{ModelType.save_dir}/latent_embeddings_all_spatial_{mode}",
    )


def balance_weight(model, adatas, save_dir, n_atlas, device):
    with open(
        f"{save_dir}/latent_embeddings_all_single_pretrain.pkl", "rb"
    ) as openfile:
        latent_embeddings_all_single = pickle.load(openfile)
    with open(
        f"{save_dir}/latent_embeddings_all_spatial_pretrain.pkl", "rb"
    ) as openfile:
        latent_embeddings_all_spatial = pickle.load(openfile)

    adatas_ = [
        ad.AnnData(
            obs=adatas[i].obs.copy(deep=False).assign(n=1),
            obsm={
                "single": latent_embeddings_all_single[i],
                "spatial": latent_embeddings_all_spatial[i],
            },
        )
        for i in range(n_atlas)
    ]

    if not os.path.exists(f"{save_dir}/ad_fusemap_single_leiden.pkl"):
        leiden_adata_single = []
        leiden_adata_spatial = []
        ad_fusemap_single_leiden = []
        ad_fusemap_spatial_leiden = []
        for adata_ in adatas_:
            sc.pp.neighbors(
                adata_,
                n_pcs=adata_.obsm["single"].shape[1],
                use_rep="single",
                metric="cosine",
            )
            sc.tl.leiden(adata_, resolution=1, key_added="fusemap_single_leiden")
            ad_fusemap_single_leiden.append(list(adata_.obs["fusemap_single_leiden"]))
            leiden_adata_single.append(
                average_embeddings(adata_, "fusemap_single_leiden", "single")
            )

            sc.pp.neighbors(
                adata_,
                n_pcs=adata_.obsm["spatial"].shape[1],
                use_rep="spatial",
                metric="cosine",
            )
            sc.tl.leiden(adata_, resolution=1, key_added="fusemap_spatial_leiden")
            ad_fusemap_spatial_leiden.append(list(adata_.obs["fusemap_spatial_leiden"]))
            leiden_adata_spatial.append(
                average_embeddings(adata_, "fusemap_spatial_leiden", "spatial")
            )

        save_obj(ad_fusemap_single_leiden, f"{save_dir}/ad_fusemap_single_leiden")
        save_obj(ad_fusemap_spatial_leiden, f"{save_dir}/ad_fusemap_spatial_leiden")
        save_obj(leiden_adata_single, f"{save_dir}/leiden_adata_single")
        save_obj(leiden_adata_spatial, f"{save_dir}/leiden_adata_spatial")

    else:
        with open(f"{save_dir}/ad_fusemap_single_leiden.pkl", "rb") as openfile:
            ad_fusemap_single_leiden = pickle.load(openfile)
        with open(f"{save_dir}/ad_fusemap_spatial_leiden.pkl", "rb") as openfile:
            ad_fusemap_spatial_leiden = pickle.load(openfile)
        try:
            with open(f"{save_dir}/leiden_adata_single.pkl", "rb") as openfile:
                leiden_adata_single = pickle.load(openfile)
            with open(f"{save_dir}/leiden_adata_spatial.pkl", "rb") as openfile:
                leiden_adata_spatial = pickle.load(openfile)
        except:
            ### need to convert
            leiden_adata_single = []
            for i in range(len(ad_fusemap_single_leiden)):
                leiden_adata_single.append(
                    sc.read_h5ad(
                        f"{save_dir}/pickle_convert/PRETRAINED_leiden_adata_single_{i}.h5ad"
                    )
                )
            leiden_adata_spatial = []
            for i in range(len(ad_fusemap_single_leiden)):
                leiden_adata_spatial.append(
                    sc.read_h5ad(
                        f"{save_dir}/pickle_convert/PRETRAINED_leiden_adata_spatial_{i}.h5ad"
                    )
                )

        for ind, adata_ in enumerate(adatas_):
            adata_.obs["fusemap_single_leiden"] = ad_fusemap_single_leiden[ind]
            adata_.obs["fusemap_spatial_leiden"] = ad_fusemap_spatial_leiden[ind]

    if len(leiden_adata_single) > 10:
        # raise ValueError('balance weight')
        balance_weight_single = get_balance_weight_subsample(
            leiden_adata_single, adatas_, "fusemap_single_leiden"
        )
        balance_weight_spatial = get_balance_weight_subsample(
            leiden_adata_spatial, adatas_, "fusemap_spatial_leiden"
        )
    else:
        balance_weight_single = get_balance_weight(
            adatas, leiden_adata_single, adatas_, "fusemap_single_leiden"
        )
        balance_weight_spatial = get_balance_weight(
            adatas, leiden_adata_spatial, adatas_, "fusemap_spatial_leiden"
        )

    balance_weight_single = [torch.tensor(i).to(device) for i in balance_weight_single]
    balance_weight_spatial = [
        torch.tensor(i).to(device) for i in balance_weight_spatial
    ]

    save_obj(balance_weight_single, f"{save_dir}/balance_weight_single")
    save_obj(balance_weight_spatial, f"{save_dir}/balance_weight_spatial")
