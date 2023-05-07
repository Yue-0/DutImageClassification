from typing import Any

from torch.nn import Module

from .repvgg import ConvNet, RepVgg
from .resnet import ResNet26, ResNet50
from .convnext import ConvNextNano, ConvNextTiny
from .swintfv2 import SwinTransformerN, SwinTransformerT

Module = Module
BASE_MODELS = [ResNet50, ConvNextTiny, SwinTransformerT]
SMALL_MODELS = [ResNet26, ConvNet, ConvNextNano, SwinTransformerN]


def model_name(name: Any) -> str:
    if name in (RepVgg, ConvNet):
        return "RepVGG"
    if name in (ResNet26, ResNet50):
        return "ResNet"
    if name in (ConvNextNano, ConvNextTiny):
        return "ConvNext"
    if name in (SwinTransformerN, SwinTransformerT):
        return "Swin Transformer"
    raise TypeError("{} is not defined".format(name))


def model_type(name: Any) -> str:
    if name is RepVgg:
        return "Rep"
    if name in BASE_MODELS:
        return "Base"
    if name in SMALL_MODELS:
        return "Small"
    raise TypeError("{} is not defined".format(name))
