import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(backbone)
        self.out_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else None
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled
