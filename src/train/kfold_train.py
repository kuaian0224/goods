import gc
import os
from typing import List

import pandas as pd
import yaml
import torch
from sklearn.model_selection import StratifiedKFold

from src.data.csv_schema import infer_columns
from src.train.train_one_fold import train_one_fold
from src.utils.io import ensure_dir, read_csv


def _attach_columns(cfg, df: pd.DataFrame):
    inferred = infer_columns(df)
    data_cfg = cfg["data"]
    data_cfg["id_col"] = data_cfg.get("id_col") or inferred["id_col"]
    data_cfg["label_col"] = data_cfg.get("label_col") or inferred["label_col"]
    data_cfg["text_cols"] = data_cfg.get("text_cols") or inferred["text_cols"]
    if not data_cfg["id_col"]:
        raise ValueError(f"Cannot infer id column from csv. Columns={list(df.columns)}")
    return data_cfg


def _assign_folds(cfg, df: pd.DataFrame) -> pd.DataFrame:
    if "fold" in df.columns:
        return df
    label_col = cfg["data"]["label_col"]
    if not label_col:
        raise ValueError("Label column missing; cannot create folds.")
    skf = StratifiedKFold(
        n_splits=cfg["train"]["folds"], shuffle=True, random_state=cfg["seed"]
    )
    df = df.copy()
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df[label_col])):
        df.loc[val_idx, "fold"] = fold
    return df


def _backup_config(cfg) -> None:
    ckpt_root = (
        cfg["train"].get("ckpt_dir")
        or cfg.get("infer", {}).get("ckpt_dir")
        or os.path.join("checkpoints", cfg["exp_name"])
    )
    path = os.path.join(ckpt_root, "config.yaml")
    ensure_dir(os.path.dirname(path))
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)


def run_kfold(cfg) -> List[dict]:
    train_path = os.path.join(cfg["data"]["root"], cfg["data"]["train_csv"])
    df = read_csv(train_path)
    _attach_columns(cfg, df)
    df = _assign_folds(cfg, df)
    _backup_config(cfg)

    fold_setting = cfg["train"].get("fold", -1)
    folds_to_run = list(range(cfg["train"]["folds"])) if fold_setting == -1 else [fold_setting]

    results = []
    for fold in folds_to_run:
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        metrics = train_one_fold(cfg, fold, train_df, val_df)
        results.append(metrics)
        # Release GPU memory between folds to avoid accumulation.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
    return results
