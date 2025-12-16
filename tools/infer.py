import argparse
import glob
import os
import sys

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


def _load_ckpts(cfg):
    pattern = os.path.join(cfg["infer"]["ckpt_dir"], "fold*", "best.pt")
    paths = sorted(glob.glob(pattern))
    if not paths and os.path.exists(os.path.join(cfg["infer"]["ckpt_dir"], "best.pt")):
        paths = [os.path.join(cfg["infer"]["ckpt_dir"], "best.pt")]
    if not paths:
        raise FileNotFoundError(f"No checkpoints found under {cfg['infer']['ckpt_dir']}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
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
    preds = probs_mean.argmax(axis=1)
    out_path = cfg["infer"]["output"]
    write_submission(out_path, ids, preds)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
