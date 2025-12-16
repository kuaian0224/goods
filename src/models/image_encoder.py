import os
import timm
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True):
        super().__init__()
        # Avoid hitting HuggingFace when a local checkpoint is already present.
        overlay = {}
        if pretrained:
            cfg = timm.models._registry.get_pretrained_cfg(backbone)
            url = getattr(cfg, "url", "")
            filename = os.path.basename(url) if url else ""
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
            local_ckpt = os.path.join(cache_dir, filename) if filename else ""
            if local_ckpt and os.path.exists(local_ckpt):
                overlay = {"file": local_ckpt, "hf_hub_id": None, "url": ""}

        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            pretrained_cfg_overlay=overlay,
        )
        self.out_dim = getattr(self.model, "num_features", None) or self.model.feature_info.num_features(-1)

    def forward(self, pixel_values):
        return self.model(pixel_values)
