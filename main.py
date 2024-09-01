import os, random, logging, warnings, numpy as np, pandas as pd

warnings.filterwarnings("ignore")
from fusemap import *
# from fusemap.spatial_integrate import *
import scanpy as sc


def main(args):
    seed_all(0)

    ### read data
    # data_pth = []
    X_input = []
    file_names = [
        args.input_data_folder_path + f
        for f in os.listdir(args.input_data_folder_path)
        if os.path.isfile(os.path.join(args.input_data_folder_path, f))
    ]

    for ind, file_name_i in enumerate(file_names):
        X = sc.read_h5ad(file_name_i)
        if "x" not in X.obs.columns:
            X.obs["x"] = X.obs["col"]
            X.obs["y"] = X.obs["row"]
        X.obs["name"] = f"section{ind}"
        X_input.append(X)
    kneighbor = ["delaunay"] * len(X_input)
    input_identity = ["ST"] * len(X_input)


    ### train model
    if args.mode == "integrate":
        spatial_integrate(X_input, args.output_save_dir, kneighbor, input_identity)
    elif args.mode == "preprocess":
        molccf_path = "/home/jialiulab/disk1/yichun/FuseMap/molCCF/"
        spatial_map(molccf_path, X_input, args.output_save_dir, kneighbor, input_identity)
    else:
        raise ValueError(f"mode {args.mode} not recognized")
    
    print("Done.")

    return


if __name__ == "__main__":
    args = parse_input_args()
    main(args)
