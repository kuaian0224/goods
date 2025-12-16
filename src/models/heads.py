import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        layers = []
        if hidden_dim and hidden_dim > 0:
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                ]
            )
        else:
            layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
