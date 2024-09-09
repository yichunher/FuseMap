Search.setIndex({"alltitles": {"1. Prepare software and data": [[17, "1.-Prepare-software-and-data"]], "2. Read data": [[17, "2.-Read-data"]], "3. Spatial integrate": [[17, "3.-Spatial-integrate"]], "API Reference": [[1, null]], "About": [[0, null]], "Authors": [[0, "authors"]], "Contents": [[2, "contents"]], "Core Functionality": [[1, "core-functionality"]], "Downloading the pretrained models": [[3, "downloading-the-pretrained-models"]], "How to Cite": [[0, "how-to-cite"]], "Impute spatial gene expression": [[15, null]], "Installation": [[3, null]], "Installing the package": [[3, "installing-the-package"]], "Integrate cells and tissues across organs and species": [[16, null]], "Integrate cells and tissues in adult mouse brain across atlases": [[17, null]], "Map cells and tissues across human/mouse or organs": [[19, null]], "Map cells and tissues to molCCF": [[18, null]], "Model Training": [[1, "model-training"]], "Quick start": [[2, "quick-start"]], "Spatial integration": [[2, "spatial-integration"]], "Spatial integration and mapping with universal gene, cell, and tissue embeddings.": [[2, "spatial-integration-and-mapping-with-universal-gene-cell-and-tissue-embeddings"]], "Spatial mapping": [[2, "spatial-mapping"]], "Step-by-step guide": [[2, "step-by-step-guide"]], "Support": [[0, "support"]], "Tutorials": [[20, null]], "Utilities": [[1, "utilities"]], "Welcome to FuseMap\u2019s documentation!": [[2, null]], "fusemap.config module": [[4, null]], "fusemap.dataset module": [[5, null]], "fusemap.logger module": [[6, null]], "fusemap.loss module": [[7, null]], "fusemap.model module": [[8, null]], "fusemap.preprocess module": [[9, null]], "fusemap.spatial_integrate module": [[10, null]], "fusemap.spatial_map module": [[11, null]], "fusemap.train module": [[12, null]], "fusemap.train_model module": [[13, null]], "fusemap.utils module": [[14, null]]}, "docnames": ["about", "api", "index", "install", "modules/config", "modules/dataset", "modules/logger", "modules/loss", "modules/model", "modules/preprocess", "modules/spatial_integrate", "modules/spatial_map", "modules/train", "modules/train_model", "modules/utils", "notebooks/spatial_impute", "notebooks/spatial_integrate_species", "notebooks/spatial_integrate_tech", "notebooks/spatial_map_mousebrain", "notebooks/spatial_map_mousehuman", "tutorials"], "envversion": {"nbsphinx": 4, "sphinx": 63, "sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1}, "filenames": ["about.rst", "api.rst", "index.rst", "install.rst", "modules/config.rst", "modules/dataset.rst", "modules/logger.rst", "modules/loss.rst", "modules/model.rst", "modules/preprocess.rst", "modules/spatial_integrate.rst", "modules/spatial_map.rst", "modules/train.rst", "modules/train_model.rst", "modules/utils.rst", "notebooks/spatial_impute.ipynb", "notebooks/spatial_integrate_species.ipynb", "notebooks/spatial_integrate_tech.ipynb", "notebooks/spatial_map_mousebrain.ipynb", "notebooks/spatial_map_mousehuman.ipynb", "tutorials.rst"], "indexentries": {}, "objects": {"fusemap": [[4, 0, 0, "-", "config"], [5, 0, 0, "-", "dataset"], [6, 0, 0, "-", "logger"], [7, 0, 0, "-", "loss"], [8, 0, 0, "-", "model"], [9, 0, 0, "-", "preprocess"], [10, 0, 0, "-", "spatial_integrate"], [11, 0, 0, "-", "spatial_map"], [12, 0, 0, "-", "train"], [13, 0, 0, "-", "train_model"], [14, 0, 0, "-", "utils"]], "fusemap.config": [[4, 1, 1, "", "FlagConfig"], [4, 1, 1, "", "ModelType"], [4, 3, 1, "", "parse_input_args"]], "fusemap.config.FlagConfig": [[4, 2, 1, "", "align_anneal"], [4, 2, 1, "", "lambda_disc_single"]], "fusemap.config.ModelType": [[4, 2, 1, "", "DIS_LAMDA"], [4, 2, 1, "", "EPS"], [4, 2, 1, "", "TRAIN_WITHOUT_EVAL"], [4, 2, 1, "", "USE_REFERENCE_PCT"], [4, 2, 1, "", "align_noise_coef"], [4, 2, 1, "", "batch_size"], [4, 2, 1, "", "dropout_rate"], [4, 2, 1, "", "hidden_dim"], [4, 2, 1, "", "lambda_ae_single"], [4, 2, 1, "", "lambda_ae_spatial"], [4, 2, 1, "", "lambda_disc_spatial"], [4, 2, 1, "", "latent_dim"], [4, 2, 1, "", "learning_rate"], [4, 2, 1, "", "lr_factor_final"], [4, 2, 1, "", "lr_factor_pretrain"], [4, 2, 1, "", "lr_limit_final"], [4, 2, 1, "", "lr_limit_pretrain"], [4, 2, 1, "", "lr_patience_final"], [4, 2, 1, "", "lr_patience_pretrain"], [4, 2, 1, "", "n_epochs"], [4, 2, 1, "", "optim_kw"], [4, 2, 1, "", "patience_limit_final"], [4, 2, 1, "", "patience_limit_pretrain"], [4, 2, 1, "", "pca_dim"], [4, 2, 1, "", "use_input"], [4, 2, 1, "", "verbose"]], "fusemap.dataset": [[5, 1, 1, "", "CustomGraphDataLoader"], [5, 1, 1, "", "CustomGraphDataset"], [5, 1, 1, "", "MapPretrainDataLoader"], [5, 1, 1, "", "MapPretrainDataset"], [5, 3, 1, "", "construct_data"], [5, 3, 1, "", "construct_mask"], [5, 3, 1, "", "get_feature_sparse"]], "fusemap.logger": [[6, 1, 1, "", "MultipleHeaderFilter"], [6, 3, 1, "", "setup_logging"]], "fusemap.logger.MultipleHeaderFilter": [[6, 4, 1, "", "filter"]], "fusemap.loss": [[7, 3, 1, "", "AE_Gene_loss"], [7, 3, 1, "", "compute_ae_loss"], [7, 3, 1, "", "compute_ae_loss_map"], [7, 3, 1, "", "compute_ae_loss_pretrain"], [7, 3, 1, "", "compute_dis_loss"], [7, 3, 1, "", "compute_dis_loss_map"], [7, 3, 1, "", "compute_dis_loss_pretrain"], [7, 3, 1, "", "compute_gene_embedding_loss"], [7, 3, 1, "", "get_balance_weight"], [7, 3, 1, "", "get_balance_weight_subsample"], [7, 3, 1, "", "prod"]], "fusemap.model": [[8, 1, 1, "", "Adj_model"], [8, 1, 1, "", "Discriminator"], [8, 1, 1, "", "FuseMapAdaptDecoder"], [8, 1, 1, "", "FuseMapDecoder"], [8, 1, 1, "", "FuseMapEncoder"], [8, 1, 1, "", "Fuse_network"], [8, 1, 1, "", "NNTransfer"], [8, 3, 1, "", "reset_parameters"]], "fusemap.model.Adj_model": [[8, 4, 1, "", "forward"]], "fusemap.model.Discriminator": [[8, 4, 1, "", "forward"]], "fusemap.model.FuseMapAdaptDecoder": [[8, 4, 1, "", "forward"]], "fusemap.model.FuseMapDecoder": [[8, 4, 1, "", "forward"]], "fusemap.model.FuseMapEncoder": [[8, 4, 1, "", "forward"]], "fusemap.model.Fuse_network": [[8, 4, 1, "", "add_adaptdecoder_module"], [8, 4, 1, "", "add_decoder_module"], [8, 4, 1, "", "add_encoder_module"]], "fusemap.model.NNTransfer": [[8, 4, 1, "", "forward"]], "fusemap.preprocess": [[9, 3, 1, "", "construct_graph"], [9, 3, 1, "", "get_allunique_gene_names"], [9, 3, 1, "", "get_spatial_input"], [9, 3, 1, "", "get_unique_gene_indices"], [9, 3, 1, "", "preprocess_adata"], [9, 3, 1, "", "preprocess_adj_sparse"], [9, 3, 1, "", "preprocess_raw"]], "fusemap.spatial_integrate": [[10, 3, 1, "", "spatial_integrate"]], "fusemap.spatial_map": [[11, 3, 1, "", "spatial_map"]], "fusemap.train": [[12, 3, 1, "", "train"]], "fusemap.train_model": [[13, 3, 1, "", "add_pretrain_to_name"], [13, 3, 1, "", "balance_weight"], [13, 3, 1, "", "get_data"], [13, 3, 1, "", "load_ref_data"], [13, 3, 1, "", "load_ref_model"], [13, 3, 1, "", "map_model"], [13, 3, 1, "", "pretrain_model"], [13, 3, 1, "", "read_model"], [13, 3, 1, "", "train_model"], [13, 3, 1, "", "transfer_weight"]], "fusemap.utils": [[14, 3, 1, "", "average_embeddings"], [14, 3, 1, "", "generate_ad_embed"], [14, 3, 1, "", "load_snapshot"], [14, 3, 1, "", "read_cell_embedding"], [14, 3, 1, "", "read_gene_embedding"], [14, 3, 1, "", "read_gene_embedding_map"], [14, 3, 1, "", "save_obj"], [14, 3, 1, "", "save_snapshot"], [14, 3, 1, "", "seed_all"], [14, 3, 1, "", "transfer_annotation"]]}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "attribute", "Python attribute"], "3": ["py", "function", "Python function"], "4": ["py", "method", "Python method"]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:attribute", "3": "py:function", "4": "py:method"}, "terms": {"": [13, 17, 20], "0": [4, 7, 8, 13], "000": 3, "0000": 7, "001": 4, "02": 4, "05": [0, 4], "1": [2, 3, 4, 7, 8, 13], "10": [0, 3, 4, 7, 8], "100": 8, "10000000000": 4, "1101": 0, "16": 4, "1e": 4, "2": [2, 4, 5, 7, 8], "20": 8, "200": 8, "2024": 0, "24": 7, "27": 0, "3": [3, 4, 7, 8], "300": 3, "4": 7, "5": [4, 7], "50": [4, 8], "512": 4, "594872": 0, "64": 4, "A": [3, 9, 10, 11, 12], "If": 6, "The": [3, 7, 8, 9, 10, 11, 12], "These": 1, "To": 0, "_": 8, "about": 2, "acceler": 3, "across": 2, "activ": 3, "adapt": 8, "adapt_model": [7, 13], "adata": [5, 7, 9, 13, 14], "adatas_": 7, "add": 8, "add_adaptdecoder_modul": 8, "add_decoder_modul": 8, "add_encoder_modul": 8, "add_pretrain_to_nam": 13, "adj": 8, "adj_al": [5, 7, 13], "adj_coo": 5, "adj_model": 8, "adjac": [7, 8], "ae_gene_loss": 7, "align_ann": 4, "align_noise_coef": 4, "all": 8, "all_unique_gen": [8, 14], "an": [4, 8], "analysi": 2, "ani": 2, "anndata": [9, 10, 11, 12, 14], "anneal": 7, "api": 2, "appropri": 6, "arg": [5, 8, 10, 11], "argpars": [10, 11], "argument": [10, 11], "atla": [0, 2], "atlas": [5, 8, 9], "atlas1": 8, "author": 2, "autoencod": 7, "averag": 14, "average_embed": 14, "balanc": 7, "balance_weight": 13, "balance_weight_single_block": 7, "balance_weight_spatial_block": 7, "base": [4, 5, 6, 8], "basic": 17, "batch_features_al": 7, "batch_siz": [4, 5, 13], "biorxiv": 0, "blocks_al": 13, "bool": 8, "brain": [0, 2, 3], "bridg": 2, "calcul": [9, 10, 11, 12, 14], "can": [0, 2, 3], "categori": [7, 14], "cd": [], "cell": [3, 7], "cell_typ": 14, "check": 2, "cite": 2, "class": [4, 5, 6, 8], "clone": [], "cluster": 2, "code": 15, "col": 5, "column": 14, "com": [], "comput": [3, 7], "compute_ae_loss": 7, "compute_ae_loss_map": 7, "compute_ae_loss_pretrain": 7, "compute_dis_loss": 7, "compute_dis_loss_map": 7, "compute_dis_loss_pretrain": 7, "compute_gene_embedding_loss": 7, "conda": 3, "config": [1, 7, 10, 11], "configur": [6, 7], "consol": 6, "consolid": 2, "construct": 5, "construct_data": 5, "construct_graph": 9, "construct_mask": 5, "contain": 14, "context": 2, "core": 2, "creat": 3, "cuda_determinist": 14, "customgraphdataload": 5, "customgraphdataset": 5, "d": 7, "data": [2, 7, 9, 10, 11, 12], "data_pth": [9, 10, 11, 12], "dataloader_pretrain_singl": 13, "dataloader_pretrain_spati": 13, "dataset": 1, "dataset_al": 5, "decod": 8, "deem": 6, "deep": 2, "delaunai": [9, 10, 11, 12], "detail": 2, "determin": 6, "devic": [5, 13], "dgl": 5, "di": 7, "dict": 7, "dimens": 8, "directori": 3, "dis_a": 7, "dis_lamda": 4, "disc": 8, "discrimin": [7, 8], "disk1": 11, "distribut": 7, "doi": 0, "download": [2, 17], "downstream": 2, "drop_last": 5, "dropout": 8, "dropout_r": [4, 8], "e": [], "each": [9, 10, 11, 12, 14], "embed": [7, 8, 14], "enc": 8, "encod": 8, "endswith": [9, 10, 11, 12], "enum": 4, "enumer": 4, "ep": 4, "epoch": 8, "epoch_fin": 14, "epoch_pretrain": 14, "estim": [3, 15], "exampl": [5, 7, 8, 9, 10, 11, 12, 14, 17], "explanatori": 20, "express": 2, "f": [9, 10, 11, 12], "factor": 7, "fals": [4, 6, 12], "familiar": 17, "featur": [5, 7], "feature_al": 13, "file": 6, "filter": 6, "final": [7, 14], "flag": 7, "flag_source_cat_singl": 7, "flag_source_cat_single_pretrain": 7, "flag_source_cat_spati": 7, "flag_source_cat_spatial_pretrain": 7, "flagconfig": [4, 7, 13], "float": [7, 8], "follow": 3, "forward": 8, "framework": 2, "from": [2, 3, 17], "function": [2, 9, 10, 11, 12, 17, 20], "fuse_network": [7, 8], "fusemap": [0, 1, 3, 17, 20], "fusemapadaptdecod": 8, "fusemapdecod": 8, "fusemapencod": 8, "g_all": [5, 13], "gb": 15, "gene": [7, 8], "gene1": 8, "gene2": 8, "gene_embed": 8, "gene_embedding_new": 8, "gene_embedding_pretrain": 8, "gene_list": 9, "gene_new": 8, "gene_pretrain": 8, "gener": 7, "generate_ad_emb": 14, "get_allunique_gene_nam": 9, "get_balance_weight": 7, "get_balance_weight_subsampl": 7, "get_data": 13, "get_feature_spars": 5, "get_spatial_input": 9, "get_unique_gene_indic": 9, "git": [], "github": [0, 3], "gpu": 3, "graph": [5, 8], "h5ad": [9, 10, 11, 12, 14], "hail": 0, "hao": 0, "he": 0, "here": 17, "hidden": 8, "hidden_dim": [4, 8], "home": 11, "hour": 3, "how": 2, "http": 0, "human": [2, 3], "i": [2, 3, 5, 6, 8], "ident": [8, 9, 10, 11, 12], "identif": 2, "ignor": 6, "import": [7, 9, 10, 11, 12], "imput": 2, "includ": [], "index": [], "indic": 8, "inform": 14, "input": [2, 8, 9, 10, 11, 12], "input_dim": 8, "input_ident": [5, 8, 9, 10, 11, 12], "instal": 2, "int": [5, 7, 8, 9], "integr": [1, 3, 10, 20], "issu": 0, "j": 5, "jia": 0, "jialiulab": 11, "k": [9, 10, 11, 12], "keep_celltyp": 14, "keep_label": 14, "keep_tissueregion": 14, "kei": [7, 8, 14], "key_leiden_categori": 7, "kneighbor": [9, 10, 11, 12], "kwarg": [5, 8], "lambda_ae_singl": 4, "lambda_ae_spati": 4, "lambda_disc_singl": 4, "lambda_disc_spati": 4, "latent": [7, 8, 14], "latent_dim": [4, 8], "layer": 8, "learn": 2, "learning_r": 4, "leiden": 7, "leiden_adata_singl": 7, "len": [9, 10, 11, 12], "list": [5, 6, 7, 8, 9, 10, 11, 12], "listdir": [9, 10, 11, 12], "liu": 0, "llm": 8, "load_ref_data": 13, "load_ref_model": 13, "load_sample_data": 7, "load_snapshot": 14, "loc": 14, "local": 3, "log": 6, "logger": 1, "loss": 1, "loss_ae_al": 7, "loss_al": 7, "lr_factor_fin": 4, "lr_factor_pretrain": 4, "lr_limit_fin": 4, "lr_limit_pretrain": 4, "lr_patience_fin": 4, "lr_patience_pretrain": 4, "mai": 6, "main": [], "manner": 3, "map": [1, 11, 20], "map_model": 13, "mappretraindataload": 5, "mappretraindataset": 5, "mask": [5, 7], "mask_batch_singl": 7, "mask_batch_spati": 7, "matric": 7, "matrix": 8, "mean": 8, "memori": 15, "merfish": 2, "messag": 6, "method": [9, 10, 11, 12], "min": 15, "mode": 13, "model": [2, 5, 7, 12, 13, 14], "modeltyp": [4, 5, 10, 11, 13], "modifi": 6, "modul": 1, "molccf": 11, "molccf_path": [11, 14], "molecular": 0, "mous": [0, 2, 3], "multipleheaderfilt": 6, "n": [3, 8], "n_atla": [5, 8, 9, 13, 14], "n_epoch": 4, "n_ob": 8, "name": [8, 14], "namespac": [10, 11], "nearest": [9, 10, 11, 12], "necessari": 3, "neighbor": [9, 10, 11, 12], "network": 8, "new": 8, "new_adata": 14, "new_train_gen": [8, 14], "nn": 7, "nntransfer": 8, "node": 8, "none": [6, 8, 9, 10, 11, 12], "norm": [4, 8], "normal": [7, 8], "notebook": 20, "num_epoch": 8, "number": [5, 7, 8, 9], "o": [9, 10, 11, 12], "ob": 14, "object": [4, 5, 9, 10, 11, 12, 14], "objt": 14, "observ": 8, "obsm": 14, "obsm_lat": 14, "optim_kw": 4, "org": 0, "organ": [2, 3], "origin": 7, "otherwis": 6, "our": [0, 2], "out": [2, 6], "output": [2, 8], "overview": 2, "packag": 2, "page": [], "panel": 2, "para": 8, "paramet": [5, 6, 7, 8, 9, 10, 11, 12, 14], "parse_input_arg": 4, "pass": 8, "path": [6, 9, 10, 11, 12, 14], "patience_limit_fin": 4, "patience_limit_pretrain": 4, "pattern": 6, "patterns_to_filt": 6, "pca": [8, 9], "pca_dim": [4, 8], "perform": 3, "phase": 7, "pip": 3, "place": 6, "pleas": 0, "preprocess": 1, "preprocess_adata": 9, "preprocess_adj_spars": 9, "preprocess_raw": 9, "preprocess_sav": 12, "pretrain": [2, 7, 8], "pretrain_index": 13, "pretrain_model": [8, 13], "pretrain_n_atla": 8, "pretrain_single_batch": 7, "pretrain_spatial_batch": 7, "pretrained_gen": [8, 14], "prod": 7, "product": 7, "provid": [1, 2, 20], "public": 0, "put": 3, "python": 3, "quick": 3, "randn": [7, 8], "rate": 8, "raw": 9, "read_cell_embed": 14, "read_gene_embed": 14, "read_gene_embedding_map": 14, "read_h5ad": [9, 10, 11, 12, 14], "read_model": 13, "recon_x": 7, "reconstruct": 7, "record": 6, "ref_dir": 13, "refer": 2, "regex": 6, "report": 0, "repres": [9, 10, 11, 12], "reset_paramet": 8, "return": [5, 6, 7, 8, 14], "rmsprop": 4, "row": 5, "sampl": 8, "sample_gene_list": 9, "sampler": 5, "save": [6, 9, 10, 11, 12], "save_dir": [12, 13, 14], "save_obj": 14, "save_path": 6, "save_snapshot": 14, "sc": [9, 10, 11, 12, 14], "scanpi": [9, 10, 11, 12], "scrna": 8, "search": [], "section": [9, 10, 11, 12, 17], "seed_al": 14, "seed_valu": 14, "seq": 2, "setup_log": 6, "sheng": 0, "shi": 0, "should": 6, "shuffl": 5, "singl": [2, 3, 7], "slide": 2, "snapshot_path": 14, "sourc": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "space": [2, 8], "spatial": [0, 1, 5, 7, 9, 10, 11, 12, 20], "spatial_dataload": 13, "spatial_dataloader_test": 13, "spatial_dataset_list": 5, "spatial_integr": [1, 2, 12], "spatial_map": [1, 2], "speci": 2, "specifi": 6, "spot": 2, "st": [9, 10, 11, 12], "str": [7, 8, 9, 10, 11, 12, 14], "string": [9, 10, 11, 12], "subsampl": 7, "support": 2, "tabl": [], "tang": 0, "task": [], "technologi": 2, "tensor": [7, 8], "thi": 17, "time": [3, 15], "tool": 2, "torch": [7, 8], "toward": 0, "tracker": 0, "train": [2, 5, 7, 8], "train_mask": [5, 13], "train_model": 1, "train_without_ev": 4, "trained_model": 13, "trained_x_num": 13, "transcriptom": 2, "transfer_annot": 14, "transfer_weight": 13, "true": [6, 14], "ttype": 14, "tutori": [2, 17], "type": [2, 7, 8, 14], "uniqu": 8, "univers": 0, "us": [0, 2, 3, 8, 9, 10, 11, 17], "use_input": [4, 5, 8, 9], "use_kei": 14, "use_llm_gene_embed": 8, "use_reference_pct": [4, 13], "user": 17, "usual": 3, "util": 2, "val_mask": [5, 13], "valid": 5, "valu": 4, "var_index": 8, "var_nam": [8, 14], "variabl": [7, 8], "varieti": [], "variou": 2, "verbos": [4, 14], "visium": 2, "wang": 0, "we": [17, 20], "weight": [7, 8], "weight_norm": 8, "wendi": 0, "where": 6, "whether": 8, "within": 2, "x": [7, 8], "x_input": [9, 10, 11, 12, 14], "xiao": 0, "xueyi": 0, "xx": [3, 15, 17], "xxx": 17, "y": 8, "yichun": [0, 11], "yichunh": [], "you": [0, 3], "z": 8, "z_distribut": 7, "z_mean": 8, "z_sampl": 8, "z_spatial": 8, "zefang": 0, "zenodo": 3, "zip": 5}, "titles": ["About", "API Reference", "Welcome to FuseMap\u2019s documentation!", "Installation", "fusemap.config module", "fusemap.dataset module", "fusemap.logger module", "fusemap.loss module", "fusemap.model module", "fusemap.preprocess module", "fusemap.spatial_integrate module", "fusemap.spatial_map module", "fusemap.train module", "fusemap.train_model module", "fusemap.utils module", "Impute spatial gene expression", "Integrate cells and tissues across organs and species", "Integrate cells and tissues in adult mouse brain across atlases", "Map cells and tissues to molCCF", "Map cells and tissues across human/mouse or organs", "Tutorials"], "titleterms": {"": 2, "1": 17, "2": 17, "3": 17, "And": [], "about": 0, "across": [16, 17, 19], "adult": 17, "api": 1, "atlas": 17, "author": 0, "brain": 17, "cell": [2, 16, 17, 18, 19], "cite": 0, "config": 4, "content": 2, "core": 1, "data": 17, "dataset": 5, "document": 2, "download": 3, "embed": 2, "exampl": [], "express": 15, "function": 1, "fusemap": [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "gene": [2, 15], "guid": 2, "how": 0, "human": 19, "imput": 15, "indic": [], "instal": 3, "integr": [2, 16, 17], "logger": 6, "loss": 7, "main": [], "map": [2, 18, 19], "model": [1, 3, 8], "modul": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "molccf": 18, "more": [], "mous": [17, 19], "organ": [16, 19], "packag": 3, "prepar": 17, "preprocess": 9, "pretrain": 3, "quick": 2, "read": 17, "refer": 1, "softwar": 17, "spatial": [2, 15, 17], "spatial_integr": 10, "spatial_map": 11, "speci": 16, "start": 2, "step": 2, "support": 0, "tabl": [], "tissu": [2, 16, 17, 18, 19], "train": [1, 12], "train_model": 13, "tutori": 20, "univers": 2, "usag": [], "util": [1, 14], "welcom": 2}})