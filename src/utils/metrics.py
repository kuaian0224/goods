import numpy as np
import torch


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float((pred == target).mean())


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int) -> float:
    if logits.numel() == 0:
        return 0.0
    _, topk = torch.topk(logits, k, dim=1)
    correct = topk.eq(target.view(-1, 1)).sum().item()
    return correct / target.size(0)
