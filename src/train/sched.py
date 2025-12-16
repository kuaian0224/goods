from transformers import get_linear_schedule_with_warmup


def build_scheduler(cfg, optimizer, num_training_steps):
    warmup_ratio = float(cfg["train"].get("warmup_ratio", 0.0))
    warmup_steps = int(num_training_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
