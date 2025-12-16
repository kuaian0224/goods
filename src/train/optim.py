import torch


def _resolve_lr(name: str, base_lr: float, lr_overrides: dict | None) -> float:
    """Pick lr for a param group with optional overrides."""
    if not lr_overrides:
        return base_lr
    if not isinstance(lr_overrides, dict):
        raise TypeError("train.lrs must be a dict when provided")
    if name in lr_overrides:
        return float(lr_overrides[name])
    if "default" in lr_overrides:
        return float(lr_overrides["default"])
    return base_lr


def _build_param_groups(model, base_lr: float, lr_overrides: dict | None):
    groups = []
    assigned = set()

    def add_group(name: str, params):
        params = [p for p in params if p.requires_grad and id(p) not in assigned]
        if not params:
            return
        lr = _resolve_lr(name, base_lr, lr_overrides)
        groups.append({"params": params, "lr": lr})
        assigned.update(id(p) for p in params)

    # Handle OpenCLIP separately to split visual/text params
    if hasattr(model, "clip_model"):
        vis_params, txt_params = [], []
        for n, p in model.clip_model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("visual."):
                vis_params.append(p)
            else:
                txt_params.append(p)
        add_group("image_encoder", vis_params)
        add_group("text_encoder", txt_params)

    if hasattr(model, "image_encoder"):
        add_group("image_encoder", model.image_encoder.parameters())
    if hasattr(model, "text_encoder"):
        add_group("text_encoder", model.text_encoder.parameters())
    if hasattr(model, "head"):
        add_group("head", model.head.parameters())

    remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in assigned]
    if remaining:
        add_group("others", remaining)

    return groups


def build_optimizer(cfg, model) -> torch.optim.Optimizer:
    base_lr = float(cfg["train"]["lr"])
    lr_overrides = cfg["train"].get("lrs")
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))

    param_groups = _build_param_groups(model, base_lr, lr_overrides)
    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
