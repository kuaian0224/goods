import argparse
import os
import random
import sys

import numpy as np

# 允许脚本以相对路径运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.csv_schema import build_text, infer_columns  # noqa: E402
from src.data.dataset import ProductDataset  # noqa: E402
from src.data.transforms import build_val_tfms  # noqa: E402
from src.utils.io import ensure_dir, load_yaml, read_csv  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    train_path = os.path.join(cfg["data"]["root"], cfg["data"]["train_csv"])
    df = read_csv(train_path)
    inferred = infer_columns(df)
    cfg["data"]["id_col"] = cfg["data"].get("id_col") or inferred["id_col"]
    cfg["data"]["label_col"] = cfg["data"].get("label_col") or inferred["label_col"]
    cfg["data"]["text_cols"] = cfg["data"].get("text_cols") or inferred["text_cols"]

    dataset = ProductDataset(
        df,
        img_dir=os.path.join(cfg["data"]["root"], cfg["data"]["train_img_dir"]),
        id_col=cfg["data"]["id_col"],
        label_col=cfg["data"]["label_col"],
        text_cols=cfg["data"]["text_cols"],
        tokenizer=None,
        tfms=build_val_tfms(cfg["data"]["img_size"]),
        is_train=False,
        multi_image_mode=cfg["data"].get("multi_image_mode", "first"),
        max_len=cfg["data"]["max_len"],
        img_size=cfg["data"]["img_size"],
    )

    n_samples = min(16, len(dataset))
    indices = random.sample(range(len(dataset)), n_samples)
    out_dir = os.path.join("outputs", "sanity")
    ensure_dir(out_dir)

    missing_images = 0
    empty_text = 0
    print("Sample records:")
    for idx in indices:
        row = dataset.df.iloc[idx]
        sid = str(row[dataset.id_col])
        text = build_text(row, dataset.text_cols)
        if text.strip() == "":
            empty_text += 1
        paths = dataset.resolve_image_paths(sid)
        if not paths:
            missing_images += 1
            img = dataset.load_image("")
        else:
            img = dataset.load_image(paths[0])
        img.save(os.path.join(out_dir, f"{sid}.jpg"))
        label_val = row[dataset.label_col] if dataset.label_col else "N/A"
        print(f"id={sid} | text={text[:60]} | label={label_val} | img={'found' if paths else 'missing'}")

    # 可选全量缺失统计，数据量大时可跳过
    if not cfg.get("fast_sanity", False):
        total_missing = sum(
            1
            for i in range(len(dataset))
            if len(dataset.resolve_image_paths(str(dataset.df.iloc[i][dataset.id_col]))) == 0
        )
        print(f"\nMissing images (all data): {total_missing}")
    else:
        print("\nMissing images (all data): skipped (fast_sanity=True)")
    print(f"Empty text in sampled {n_samples}: {empty_text}")
    print(f"Saved sample images to {out_dir}")


if __name__ == "__main__":
    main()
