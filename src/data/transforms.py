from typing import Callable

from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_train_tfms(img_size: int) -> Callable:
    return T.Compose(
        [
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_val_tfms(img_size: int) -> Callable:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_clip_train_tfms(img_size: int) -> Callable:
    return T.Compose(
        [
            T.Resize((img_size + 32, img_size + 32), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def build_clip_val_tfms(img_size: int) -> Callable:
    return T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
