import argparse
import os
import sys

from sklearn.model_selection import StratifiedKFold

# 允许脚本以相对路径运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.csv_schema import infer_columns  # noqa: E402
from src.utils.io import load_yaml, read_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_path = os.path.join(cfg["data"]["root"], cfg["data"]["train_csv"])
    df = read_csv(train_path)
    inferred = infer_columns(df)
    label_col = cfg["data"].get("label_col") or inferred["label_col"]
    if not label_col:
        raise ValueError("Cannot infer label column for fold split.")

    skf = StratifiedKFold(n_splits=cfg["train"]["folds"], shuffle=True, random_state=cfg["seed"])
    df = df.copy()
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df[label_col])):
        df.loc[val_idx, "fold"] = fold
    out_path = os.path.join(cfg["data"]["root"], "train_folds.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved folds to {out_path}")


if __name__ == "__main__":
    main()
