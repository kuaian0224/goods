import os
from typing import Tuple

import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .dataset import ProductDataset
from .transforms import (
    build_clip_train_tfms,
    build_clip_val_tfms,
    build_train_tfms,
    build_val_tfms,
)


def build_tokenizer(cfg) -> PreTrainedTokenizerBase | None:
    model_type = cfg["model"]["type"]
    if model_type == "image":
        return None
    if model_type == "clip_fusion":
        import open_clip

        clip_backbone = cfg["model"].get("clip_backbone", "ViT-L-14")
        return open_clip.get_tokenizer(clip_backbone)
    tokenizer_name = cfg["model"].get("text_backbone", "bert-base-multilingual-cased")
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)



def _img_dir(cfg, is_train: bool) -> str:
    root = cfg["data"]["root"]
    subdir = cfg["data"]["train_img_dir"] if is_train else cfg["data"]["test_img_dir"]
    return os.path.join(root, subdir)


def build_dataloaders(cfg, train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer=None) -> Tuple[DataLoader, DataLoader]:
    img_size = cfg["data"]["img_size"]
    if cfg["model"]["type"] == "clip_fusion":
        train_tfms = build_clip_train_tfms(img_size)
        val_tfms = build_clip_val_tfms(img_size)
    else:
        train_tfms = build_train_tfms(img_size)
        val_tfms = build_val_tfms(img_size)

    train_dataset = ProductDataset(
        train_df,
        img_dir=_img_dir(cfg, is_train=True),
        id_col=cfg["data"].get("id_col"),
        label_col=cfg["data"].get("label_col"),
        text_cols=cfg["data"].get("text_cols"),
        tokenizer=tokenizer,
        tfms=train_tfms,
        is_train=True,
        multi_image_mode=cfg["data"].get("multi_image_mode", "first"),
        max_len=cfg["data"]["max_len"],
        img_size=img_size,
    )

    val_dataset = ProductDataset(
        val_df,
        img_dir=_img_dir(cfg, is_train=True),
        id_col=cfg["data"].get("id_col"),
        label_col=cfg["data"].get("label_col"),
        text_cols=cfg["data"].get("text_cols"),
        tokenizer=tokenizer,
        tfms=val_tfms,
        is_train=True,
        multi_image_mode=cfg["data"].get("multi_image_mode", "first"),
        max_len=cfg["data"]["max_len"],
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def build_test_loader(cfg, test_df: pd.DataFrame, tokenizer=None) -> DataLoader:
    img_size = cfg["data"]["img_size"]
    if cfg["model"]["type"] == "clip_fusion":
        val_tfms = build_clip_val_tfms(img_size)
    else:
        val_tfms = build_val_tfms(img_size)
    test_dataset = ProductDataset(
        test_df,
        img_dir=_img_dir(cfg, is_train=False),
        id_col=cfg["data"].get("id_col"),
        label_col=None,
        text_cols=cfg["data"].get("text_cols"),
        tokenizer=tokenizer,
        tfms=val_tfms,
        is_train=False,
        multi_image_mode=cfg["data"].get("multi_image_mode", "first"),
        max_len=cfg["data"]["max_len"],
        img_size=img_size,
    )
    return DataLoader(
        test_dataset,
        batch_size=cfg["infer"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
