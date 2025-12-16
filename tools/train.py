import argparse
import os
import sys

# 允许脚本以相对路径运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.train.kfold_train import run_kfold  # noqa: E402
from src.utils.io import load_yaml  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--fold", type=int, default=None, help="Override fold index (-1 for all)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.fold is not None:
        cfg["train"]["fold"] = args.fold

    results = run_kfold(cfg)
    if results:
        avg_acc = sum(r["best_acc"] for r in results) / len(results)
        print(f"Finished folds. Avg best acc: {avg_acc:.4f}")
        for r in results:
            print(f"Fold {r['fold']}: best_acc={r['best_acc']:.4f}")


if __name__ == "__main__":
    main()
