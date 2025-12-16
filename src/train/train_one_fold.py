import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.datamodule import build_dataloaders, build_tokenizer
from src.models.factory import build_model
from src.utils.io import ensure_dir
from src.utils.metrics import topk_accuracy
from src.utils.seed import set_seed
from .optim import build_optimizer
from .sched import build_scheduler


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _class_weights(cfg, train_df: pd.DataFrame) -> torch.Tensor | None:
    if not cfg["train"].get("class_weight") or cfg["data"].get("label_col") is None:
        return None
    num_classes = cfg["data"]["num_classes"]
    counts = train_df[cfg["data"]["label_col"]].value_counts().to_dict()
    weights = torch.ones(num_classes, dtype=torch.float)
    total = sum(counts.values())
    for cls, cnt in counts.items():
        if int(cls) < num_classes:
            weights[int(cls)] = total / (cnt + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights


def _save_checkpoint(model, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save({"model": model.state_dict()}, path)


def train_one_fold(cfg, fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, float]:
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg["seed"])

    tokenizer = build_tokenizer(cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_df, val_df, tokenizer)

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)

    grad_accum = cfg["train"].get("grad_accum", 1)
    total_steps = math.ceil(len(train_loader) * cfg["train"]["epochs"] / grad_accum)
    scheduler = build_scheduler(cfg, optimizer, total_steps)
    scaler = GradScaler(enabled=cfg["train"].get("amp", False))

    class_weight = _class_weights(cfg, train_df)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weight,
        label_smoothing=cfg["train"].get("label_smoothing", 0.0),
    )

    log_dir = os.path.join("logs", cfg["exp_name"])
    ensure_dir(log_dir)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"fold{fold}_tb"))
    csv_log_path = os.path.join(log_dir, f"fold{fold}.csv")
    ckpt_root = (
        cfg["train"].get("ckpt_dir")
        or cfg.get("infer", {}).get("ckpt_dir")
        or os.path.join("checkpoints", cfg["exp_name"])
    )
    ckpt_path = os.path.join(ckpt_root, f"fold{fold}", "best.pt")

    best_acc = 0.0
    epochs_no_improve = 0
    log_rows = []

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}", ncols=100)

        for step, batch in enumerate(pbar):
            batch = _to_device(batch, device)
            use_amp = cfg["train"].get("amp", False)
            with autocast(device_type="cuda", enabled=use_amp) if device.type == "cuda" else autocast(
                device_type="cpu", enabled=use_amp
            ):
                logits = model(batch)
                loss = criterion(logits, batch["labels"])
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if getattr(optimizer, "_step_count", 0) > 0:
                    scheduler.step()

            running_loss += loss.item() * grad_accum
            if (step + 1) % cfg["train"].get("log_every", 50) == 0:
                pbar.set_postfix({"loss": f"{(running_loss/(step+1)):.4f}"})

        # validation
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, device)
                logits = model(batch)
                all_logits.append(logits.cpu())
                all_labels.append(batch["labels"].cpu())

        val_logits = torch.cat(all_logits, dim=0)
        val_labels = torch.cat(all_labels, dim=0)
        preds = val_logits.argmax(dim=1)
        val_acc = float((preds == val_labels).float().mean().item())
        val_top3 = topk_accuracy(val_logits, val_labels, k=3)

        writer.add_scalar("loss/train", running_loss / len(train_loader), epoch)
        writer.add_scalar("acc/val_top1", val_acc, epoch)
        writer.add_scalar("acc/val_top3", val_top3, epoch)

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(len(train_loader), 1),
                "val_acc": val_acc,
                "val_top3": val_top3,
            }
        )

        # save CSV log
        pd.DataFrame(log_rows).to_csv(csv_log_path, index=False)

        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            epochs_no_improve = 0
            _save_checkpoint(model, ckpt_path)
        else:
            epochs_no_improve += 1

        if cfg["train"].get("early_stop_patience") is not None and epochs_no_improve > cfg["train"].get(
            "early_stop_patience", 1000
        ):
            break

    writer.close()
    return {"fold": fold, "best_acc": best_acc}
