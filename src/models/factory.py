from .fusion_model import ClipFusionModel, FusionModel, ImageOnlyModel, TextOnlyModel


def build_model(cfg):
    model_cfg = cfg["model"]
    num_classes = cfg["data"]["num_classes"]
    dropout = model_cfg.get("dropout", 0.1)
    hidden_dim = model_cfg.get("hidden_dim", 0)

    model_type = model_cfg["type"]
    if model_type == "image":
        return ImageOnlyModel(
            backbone=model_cfg["image_backbone"],
            pretrained=model_cfg.get("image_pretrained", True),
            num_classes=num_classes,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
    if model_type == "text":
        return TextOnlyModel(
            backbone=model_cfg["text_backbone"],
            num_classes=num_classes,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
    if model_type == "fusion":
        return FusionModel(
            image_backbone=model_cfg["image_backbone"],
            text_backbone=model_cfg["text_backbone"],
            pretrained=model_cfg.get("image_pretrained", True),
            fusion=model_cfg.get("fusion", "concat"),
            num_classes=num_classes,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
    if model_type == "clip_fusion":
        return ClipFusionModel(
            clip_backbone=model_cfg.get("clip_backbone", "ViT-L-14"),
            clip_pretrained=model_cfg.get("clip_pretrained", "openai"),
            num_classes=num_classes,
            dropout=dropout,
            hidden_dim=hidden_dim,
            trainable=model_cfg.get("clip_trainable", False),
            force_quick_gelu=model_cfg.get("clip_force_quick_gelu", False),
        )
    raise ValueError(f"Unsupported model type: {model_type}")
