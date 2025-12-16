import argparse
import glob
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

# 允许脚本以相对路径运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.csv_schema import infer_columns  # noqa: E402
from src.data.datamodule import build_test_loader, build_tokenizer  # noqa: E402
from src.infer.ensemble import ensemble_mean  # noqa: E402
from src.infer.predict import predict_proba  # noqa: E402
from src.infer.submission import write_submission  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.utils.io import load_yaml, read_csv  # noqa: E402


def _load_ckpts(cfg) -> List[str]:
    pattern = os.path.join(cfg["infer"]["ckpt_dir"], "fold*", "best.pt")
    paths = sorted(glob.glob(pattern))
    if not paths and os.path.exists(os.path.join(cfg["infer"]["ckpt_dir"], "best.pt")):
        paths = [os.path.join(cfg["infer"]["ckpt_dir"], "best.pt")]
    if not paths:
        raise FileNotFoundError(f"No checkpoints found under {cfg['infer']['ckpt_dir']}")
    return paths


def _infer_single_cfg(cfg_path: str) -> Tuple[List[str], np.ndarray]:
    cfg = load_yaml(cfg_path)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    test_path = os.path.join(cfg["data"]["root"], cfg["data"]["test_csv"])
    test_df = read_csv(test_path)
    inferred = infer_columns(test_df)
    cfg["data"]["id_col"] = cfg["data"].get("id_col") or inferred["id_col"]
    cfg["data"]["text_cols"] = cfg["data"].get("text_cols") or inferred["text_cols"]

    tokenizer = build_tokenizer(cfg)
    test_loader = build_test_loader(cfg, test_df, tokenizer)

    ckpts = _load_ckpts(cfg)
    probs_list = []
    ids = None

    for ckpt_path in ckpts:
        model = build_model(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        fold_ids, probs = predict_proba(cfg, model, test_loader)
        probs_list.append(probs)
        ids = fold_ids

    probs_mean = ensemble_mean(probs_list)
    return ids, probs_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True, help="List of config yaml paths to ensemble")
    parser.add_argument("--weights", nargs="*", type=float, default=None, help="Optional weights per config")
    parser.add_argument("--output", default="submission_ensemble.csv", help="Output submission path")
    args = parser.parse_args()

    num_cfgs = len(args.configs)
    weights = args.weights if args.weights else [1.0] * num_cfgs
    if len(weights) != num_cfgs:
        raise ValueError("weights length must match configs length")
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    ids_master: List[str] | None = None
    merged_probs: np.ndarray | None = None

    for cfg_path, w in zip(args.configs, weights):
        ids, probs = _infer_single_cfg(cfg_path)
        if ids_master is None:
            ids_master = ids
            merged_probs = probs * w
        else:
            if ids != ids_master:
                raise RuntimeError(f"ID order mismatch for config {cfg_path}")
            merged_probs = merged_probs + probs * w  # type: ignore[arg-type]

    assert ids_master is not None and merged_probs is not None
    preds = merged_probs.argmax(axis=1)
    write_submission(args.output, ids_master, preds)
    print(f"Saved ensemble submission to {args.output}")


if __name__ == "__main__":
    main()
