from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def _to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def predict_proba(cfg, model, loader) -> Tuple[List[str], np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    ids: List[str] = []
    probs_list = []
    tta = cfg["infer"].get("tta", False) and cfg["model"]["type"] != "text"

    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer", ncols=100):
            ids.extend(batch["id"])
            batch_dev = _to_device(batch, device)
            logits = model(batch_dev)

            if tta:
                flipped = torch.flip(batch_dev["pixel_values"], dims=[3])
                batch_dev_flip = dict(batch_dev)
                batch_dev_flip["pixel_values"] = flipped
                logits_flip = model(batch_dev_flip)
                logits = (logits + logits_flip) / 2

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)

    return ids, np.concatenate(probs_list, axis=0)
