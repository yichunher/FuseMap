import sys
import argparse
from typing import Dict, Any
from enum import Enum

def parse_input_args():
    parser = argparse.ArgumentParser(description="FuseMap")

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/jialiulab/disk1/yichun/FuseMap/sample_data/",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/jialiulab/disk1/yichun/FuseMap/output/sample_data1/",
    )

    args = parser.parse_args()

    return args

class FlagConfig:
    lambda_disc_single = 1
    align_anneal = 1e10



class ModelType(Enum):
    pca_dim=50
    hidden_dim=512
    latent_dim=64
    dropout_rate=0.2
    n_epochs=20
    batch_size=16
    learning_rate=0.001
    optim_kw="RMSprop"
    use_input="norm"
    harmonized_gene=True
    # data_pth=None
    # preprocess_save=False
    # lambda_disc_single=1
    lambda_ae_single=1
    lambda_disc_spatial=1
    lambda_ae_spatial=1
    align_noise_coef=1.5
    # align_anneal=1e10
    lr_patience_pretrain=2
    lr_factor_pretrain=0.5
    lr_limit_pretrain=0.00001
    patience_limit_final=5
    lr_patience_final=3
    lr_factor_final=0.5
    lr_limit_final=0.00001
    patience_limit_pretrain=4
    EPS=1e-10
    DIS_LAMDA=2
    TRAIN_WITHOUT_EVAL=3
    USE_REFERENCE_PCT=0.25
    verbose=False

    # PCA_DIM=50
    # HIDDEN_DIM=512
    # LATENT_DIM=64
    # DROPOUT_RATE=0.2
    # N_EPOCHS=200
    # BATCH_SIZE=16
    # LEARNING_RATE=0.001
    # OPTIM_KW="RMSprop"
    # USE_INPUT="norm"
    # HARMONIZED_GENE=True
    # DATA_PTH=None
    # PREPROCESS_SAVE=False
    # LAMBDA_DISC_SINGLE=1
    # LAMBDA_AE_SINGLE=1
    # LAMBDA_DISC_SPATIAL=1
    # LAMBDA_AE_SPATIAL=1
    # ALIGN_NOISE_COEF=1.5
    # ALIGN_ANNEAL=1e10
    # LR_PATIENCE_PRETRAIN=2
    # LR_FACTOR_PRETRAIN=0.5
    # LR_LIMIT_PRETRAIN=0.00001
    # PATIENCE_LIMIT_FINAL=5
    # LR_PATIENCE_FINAL=3
    # LR_FACTOR_FINAL=0.5
    # LR_LIMIT_FINAL=0.00001
    # PATIENCE_LIMIT_PRETRAIN=4
    # EPS=1e-10
    # DIS_LAMDA=2
    # TRAIN_WITHOUT_EVAL=3
    # USE_REFERENCE_PCT=0.25
    # VERBOSE=False

# def parse_args():
#     if "ipykernel_launcher" in sys.argv[0]:
#         # Default values when running in Jupyter notebook
#         args = argparse.Namespace(
#             pca_dim=50,
#             hidden_dim=512,
#             latent_dim=64,
#             dropout_rate=0.2,
#             n_epochs=200,
#             batch_size=16,
#             learning_rate=0.001,
#             optim_kw="RMSprop",
#             use_input="norm",
#             harmonized_gene=True,
#             data_pth=None,
#             preprocess_save=False,
#             lambda_disc_single=1,
#             lambda_ae_single=1,
#             lambda_disc_spatial=1,
#             lambda_ae_spatial=1,
#             align_noise_coef=1.5,
#             align_anneal=1e10,
#             lr_patience_pretrain=2,
#             lr_factor_pretrain=0.5,
#             lr_limit_pretrain=0.00001,
#             patience_limit_final=5,
#             lr_patience_final=3,
#             lr_factor_final=0.5,
#             lr_limit_final=0.00001,
#             patience_limit_pretrain=4,
#             EPS=1e-10,
#             DIS_LAMDA=2,
#             TRAIN_WITHOUT_EVAL=3,
#             USE_REFERENCE_PCT=0.25,
#             verbose=False,
#         )
#     else:
#         # Argument parsing when running from the command line
#         parser = argparse.ArgumentParser(description="FuseMap")
#         parser.add_argument("--pca_dim", type=int, default=50)
#         parser.add_argument("--hidden_dim", type=int, default=512)
#         parser.add_argument("--latent_dim", type=int, default=64)
#         parser.add_argument("--dropout_rate", type=float, default=0.2)
#         parser.add_argument("--n_epochs", type=int, default=200)
#         parser.add_argument("--batch_size", type=int, default=16)
#         parser.add_argument("--learning_rate", type=float, default=0.001)
#         parser.add_argument("--optim_kw", type=str, default="RMSprop")
#         parser.add_argument("--use_input", type=str, default="norm")
#         parser.add_argument("--harmonized_gene", type=bool, default=True)
#         parser.add_argument("--data_pth", type=str, default=None)
#         parser.add_argument("--preprocess_save", type=bool, default=False)
#         parser.add_argument("--lambda_disc_single", type=int, default=1)
#         parser.add_argument("--lambda_ae_single", type=int, default=1)
#         parser.add_argument("--lambda_disc_spatial", type=int, default=1)
#         parser.add_argument("--lambda_ae_spatial", type=int, default=1)
#         parser.add_argument("--align_noise_coef", type=float, default=1.5)
#         parser.add_argument("--align_anneal", type=float, default=1e10)
#         parser.add_argument("--lr_patience_pretrain", type=int, default=2)
#         parser.add_argument("--lr_factor_pretrain", type=float, default=0.5)
#         parser.add_argument("--lr_limit_pretrain", type=float, default=0.00001)
#         parser.add_argument("--patience_limit_final", type=int, default=5)
#         parser.add_argument("--lr_patience_final", type=int, default=3)
#         parser.add_argument("--lr_factor_final", type=float, default=0.5)
#         parser.add_argument("--lr_limit_final", type=float, default=0.00001)
#         parser.add_argument("--patience_limit_pretrain", type=int, default=4)
#         parser.add_argument("--EPS", type=float, default=1e-10)
#         parser.add_argument("--DIS_LAMDA", type=int, default=2)
#         parser.add_argument("--TRAIN_WITHOUT_EVAL", type=int, default=3)
#         parser.add_argument("--USE_REFERENCE_PCT", type=float, default=0.25)
#         parser.add_argument("--verbose", type=bool, default=False)
#         args = parser.parse_args()

#     return args
