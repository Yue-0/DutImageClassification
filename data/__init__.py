import os
import random
from glob import glob
from typing import Union

import torch
import torchvision as tv
from torch.utils.data import DataLoader

__all__ = ["Dataset", "DataLoader"]
__author__ = "YueLin"

LabelType = Union[torch.Tensor, int, str]
DataType = tuple[torch.Tensor, LabelType]


class Dataset:
    def __init__(self,
                 path: str,
                 mode: str,
                 test_size: int = 160,
                 train_size: int = 128,
                 label_smooth: float = 0.1,
                 color_transform: float = 0.1):
        images = "*.jpg"
        self.mode = mode
        self.epsilon = label_smooth
        assert mode in ("train", "val", "test")
        if mode != "test":
            images = os.path.join('*', images)
        self.path = path
        self.size = train_size
        self.data = glob(os.path.join(path, mode, images))
        self.resize = tv.transforms.Resize(test_size, antialias=True)
        if mode == "train":
            p = color_transform
            self.color = tv.transforms.Compose([
                tv.transforms.ColorJitter(p, p),
                tv.transforms.RandomGrayscale(p)
            ])
            self.transform = tv.transforms.Compose([
                tv.transforms.RandomCrop(train_size),
                tv.transforms.RandomHorizontalFlip()
            ])
        else:
            self.color = None
            self.transform = tv.transforms.TenCrop(train_size)
        self.style = "original"
        self.styles = ("original", "anime", "sketch", "van")
        self.augments = ("MixUp", "CutOut", "CutMix", "ColorTransform")
        self.num_classes = len(os.listdir(os.path.join(path, "train")))
        self.normalization = tv.transforms.Normalize((127.5,) * 3, (127.5,) * 3)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        if self.mode == "val":
            for item in range(len(self)):
                images, label = [], None
                for style in self.styles:
                    self.style = style
                    image, label = self[item]
                    images.append(image)
                yield torch.cat(images, 0), label
        else:
            for item in range(len(self)):
                yield self[item]

    def __getitem__(self, item: int) -> DataType:
        if self.mode == "train":
            self.style = random.choice(self.styles)
        return self.augment(*self.read(self.data[item]))

    def read(self, path: str) -> DataType:
        image = tv.io.read_image(self.convert(path))
        if image.shape[0] == 1:
            image = image.expand(3, *image.shape[1:])
        for transform in (self.resize, self.transform):
            image = transform(image)
        if self.mode != "train":
            image = torch.stack(image)
        if self.mode == "test":
            label = os.path.split(path)[1]
        else:
            label = int(os.path.split(os.path.split(path)[0])[1])
        return image.to(torch.float32), self.onehot(label)

    def onehot(self, label: Union[int, str]) -> LabelType:
        if self.mode == "train":
            onehot = torch.ones(self.num_classes)
            onehot *= self.epsilon / (self.num_classes - 1)
            onehot[label] = 1 - self.epsilon
            return onehot
        return label

    def convert(self, path: str) -> str:
        if self.style == "original" or self.mode == "test":
            return path
        path, jpg = os.path.split(path)
        path, cls = os.path.split(path)
        path, mode = os.path.split(path)
        return os.path.join(path, self.style, mode, cls, jpg)

    def augment(self, image: torch.Tensor, label: LabelType) -> DataType:
        if self.mode == "train" and (p := random.random()) < 0.5:
            augment = random.choice(self.augments)
            if augment.startswith("Cut"):
                w, h = [random.randint(0, self.size // 3) for _ in range(2)]
                x, y = [random.randint(0, self.size - size) for size in (w, h)]
                if augment.endswith("Out"):
                    image[:, y:y + h, x:x + w] = 0
                else:
                    p = w * h / self.size ** 2
                    image0, label0 = self.read(random.choice(self.data))
                    image[:, y:y + h, x:x + w] = image0[:, y:y + h, x:x + w]
                    label = p * label0 + (1 - p) * label
            elif augment.startswith("Color"):
                if self.style != "sketch":
                    image = self.color(image)
            else:
                image0, label0 = self.read(random.choice(self.data))
                image = (1 - p) * image + p * image0
                label = (1 - p) * label + p * label0
        return self.normalization(image), label
