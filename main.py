import os, random, logging, warnings, numpy as np, pandas as pd

warnings.filterwarnings("ignore")

from fusemap.train import *
import scanpy as sc


def main(args):
    seed_all(0)
    data_pth = []
    X_input = []
    file_names = [
        args.data_path + f
        for f in os.listdir(args.data_path)
        if os.path.isfile(os.path.join(args.data_path, f))
    ]

    for ind, file_name_i in enumerate(file_names):
        X = sc.read_h5ad(file_name_i)
        if "x" not in X.obs.columns:
            X.obs["x"] = X.obs["col"]
            X.obs["y"] = X.obs["row"]
        data_pth.append(file_name_i)
        X.obs["name"] = f"section{ind}"
        X_input.append(X)
    kneighbor = ["delaunay"] * len(X_input)
    input_identity = ["ST"] * len(X_input)

    train(X_input, args.save_dir, kneighbor, input_identity, data_pth=data_pth)

    pass


if __name__ == "__main__":
    args = parse_input_args()
    main(args)
