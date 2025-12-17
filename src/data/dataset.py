import glob
import os
import random
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import PreTrainedTokenizerBase

from .csv_schema import build_text, infer_columns


class ProductDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        id_col: Optional[str] = None,
        label_col: Optional[str] = None,
        text_cols: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        tfms: Optional[Callable] = None,
        is_train: bool = True,
        multi_image_mode: str = "first",
        max_len: int = 64,
        img_size: int = 224,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        inferred = infer_columns(df)
        self.id_col = id_col or inferred["id_col"]
        if not self.id_col:
            raise ValueError(f"Cannot find id column in df columns={list(df.columns)}")
        self.label_col = label_col or inferred["label_col"]
        self.text_cols = text_cols if text_cols is not None else inferred["text_cols"]
        self.tokenizer = tokenizer
        self.tfms = tfms or T.ToTensor()
        self.is_train = is_train
        self.multi_image_mode = multi_image_mode
        self.max_len = max_len
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def resolve_image_paths(self, sample_id: str) -> List[str]:
        patterns = [
            f"{sample_id}.jpg",
            f"{sample_id}.png",
            f"{sample_id}.jpeg",
            f"{sample_id}.webp",
            f"{sample_id}_*.*",
        ]
        paths: List[str] = []
        for pat in patterns:
            globbed = glob.glob(os.path.join(self.img_dir, pat))
            paths.extend(globbed)
        unique_sorted = sorted(list(set(paths)))
        return unique_sorted

    def load_image(self, path: str) -> Image.Image:
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception:
            return Image.new("RGB", (self.img_size, self.img_size), color=0)

    def _apply_tfms(self, img: Image.Image) -> torch.Tensor:
        if self.tfms:
            return self.tfms(img)
        return T.ToTensor()(img)

    def _choose_image_tensor(self, sample_id: str) -> torch.Tensor:
        paths = self.resolve_image_paths(sample_id)
        if not paths:
            img = Image.new("RGB", (self.img_size, self.img_size), color=0)
            return self._apply_tfms(img)

        if self.multi_image_mode == "random" and len(paths) > 1 and self.is_train:
            path = random.choice(paths)
            return self._apply_tfms(self.load_image(path))

        if self.multi_image_mode == "mean_pool" and len(paths) > 1:
            tensors = [self._apply_tfms(self.load_image(p)) for p in paths]
            return torch.stack(tensors, dim=0).mean(dim=0)

        path = paths[0]
        return self._apply_tfms(self.load_image(path))

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None or text is None:
            return {
                "input_ids": torch.zeros(self.max_len, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_len, dtype=torch.long),
            }

        # open_clip tokenizer is a callable that returns tensor tokens directly.
        if callable(self.tokenizer) and not hasattr(self.tokenizer, "encode_plus"):
            tokens = self.tokenizer([text])
            if hasattr(tokens, "squeeze"):
                tokens = tokens.squeeze(0)
            attn = (tokens != 0).long()
            return {
                "input_ids": tokens,
                "attention_mask": attn,
            }

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample_id = str(row[self.id_col])

        text = build_text(row, self.text_cols)
        tokens = self._tokenize_text(text)
        pixel_values = self._choose_image_tensor(sample_id)

        item: Dict[str, torch.Tensor] = {
            "id": sample_id,
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

        if self.is_train and self.label_col:
            label = int(row[self.label_col])
            item["labels"] = torch.tensor(label, dtype=torch.long)
        return item
