import torch
import torch.nn as nn

try:
    import open_clip
except ImportError:  # pragma: no cover
    open_clip = None

from .heads import ClassificationHead
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class ImageOnlyModel(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, num_classes: int, dropout: float, hidden_dim: int):
        super().__init__()
        self.image_encoder = ImageEncoder(backbone, pretrained=pretrained)
        self.head = ClassificationHead(
            in_dim=self.image_encoder.out_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(self, batch) -> torch.Tensor:
        img_feat = self.image_encoder(batch["pixel_values"])
        return self.head(img_feat)


class TextOnlyModel(nn.Module):
    def __init__(self, backbone: str, num_classes: int, dropout: float, hidden_dim: int):
        super().__init__()
        self.text_encoder = TextEncoder(backbone)
        self.head = ClassificationHead(
            in_dim=self.text_encoder.out_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(self, batch) -> torch.Tensor:
        txt_feat = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        return self.head(txt_feat)


class FusionModel(nn.Module):
    def __init__(
        self,
        image_backbone: str,
        text_backbone: str,
        pretrained: bool,
        fusion: str,
        num_classes: int,
        dropout: float,
        hidden_dim: int,
    ):
        super().__init__()
        self.fusion = fusion
        self.image_encoder = ImageEncoder(image_backbone, pretrained=pretrained)
        self.text_encoder = TextEncoder(text_backbone)

        img_dim = self.image_encoder.out_dim
        txt_dim = self.text_encoder.out_dim

        if fusion == "sum":
            if img_dim != txt_dim:
                self.img_proj = nn.Linear(img_dim, txt_dim)
                img_dim = txt_dim
            else:
                self.img_proj = nn.Identity()
            head_in = txt_dim
        else:
            self.img_proj = None
            head_in = img_dim + txt_dim

        if fusion == "gated_concat":
            self.gate = nn.Sequential(nn.Linear(head_in, head_in), nn.Sigmoid())
        else:
            self.gate = None

        self.head = ClassificationHead(in_dim=head_in, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)

    def _fuse(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        if self.fusion == "sum":
            img_feat = self.img_proj(img_feat)
            fused = img_feat + txt_feat
        else:
            fused = torch.cat([img_feat, txt_feat], dim=1)
            if self.fusion == "gated_concat" and self.gate is not None:
                fused = fused * self.gate(fused)
        return fused

    def forward(self, batch) -> torch.Tensor:
        img_feat = self.image_encoder(batch["pixel_values"])
        txt_feat = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        fused = self._fuse(img_feat, txt_feat)
        return self.head(fused)


class ClipFusionModel(nn.Module):
    def __init__(
        self,
        clip_backbone: str,
        clip_pretrained: str,
        num_classes: int,
        dropout: float,
        hidden_dim: int,
        trainable: bool = False,
        force_quick_gelu: bool = False,
    ):
        super().__init__()
        if open_clip is None:
            raise ImportError("open_clip is required for ClipFusionModel. Please install open-clip-torch.")
        self.clip_model = open_clip.create_model(
            clip_backbone,
            pretrained=clip_pretrained,
            force_quick_gelu=force_quick_gelu,
        )
        self.frozen = not trainable
        if self.frozen:
            for p in self.clip_model.parameters():
                p.requires_grad = False
        embed_dim = self.clip_model.text_projection.shape[1]
        img_dim = getattr(self.clip_model.visual, "output_dim", embed_dim)
        head_in = img_dim + embed_dim
        self.head = ClassificationHead(in_dim=head_in, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, batch) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                img_feat = self.clip_model.encode_image(batch["pixel_values"])
                txt_feat = self.clip_model.encode_text(batch["input_ids"])
        else:
            img_feat = self.clip_model.encode_image(batch["pixel_values"])
            txt_feat = self.clip_model.encode_text(batch["input_ids"])

        img_feat = img_feat.float()
        txt_feat = txt_feat.float()
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.head(fused)

    @property
    def image_encoder(self):
        return self.clip_model.visual

    @property
    def text_encoder(self):
        return self.clip_model
