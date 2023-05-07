from torch import nn
from torchvision.models import resnet

__all__ = ["ResNet26", "ResNet50"]
__author__ = "YueLin"


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 down_sample: bool):
        stride = 2 if down_sample else 1
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if down_sample else None

    def forward(self, x: resnet.Tensor) -> resnet.Tensor:
        y = self.block(x)
        if self.down_sample:
            x = self.down_sample(x)
        return self.relu(x + y)


class Res3Blocks(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 down_sample: bool):
        super(Res3Blocks, self).__init__()
        self.res = nn.Sequential(
            ResBlock(in_channels, out_channels, down_sample),
            ResBlock(out_channels, out_channels, False),
            ResBlock(out_channels, out_channels, False)
        )

    def forward(self, x: resnet.Tensor) -> resnet.Tensor:
        return self.res(x)


class ResNet26(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet26, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1),
            Res3Blocks(64, 64, False),
            Res3Blocks(64, 128, True),
            Res3Blocks(128, 256, True),
            Res3Blocks(256, 256, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: resnet.Tensor) -> resnet.Tensor:
        return self.fc(self.conv(x))


def resnet50(num_classes: int) -> resnet.ResNet:
    net = resnet.resnet50(
        weights=resnet.ResNet50_Weights.DEFAULT
    )
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


ResNet50 = resnet50


if __name__ == "__main__":
    from torch import randn
    from thop import profile
    model = ResNet26(50)
    inputs = randn(1, 3, 128, 128)
    flops, params = profile(model, (inputs,))
    print("{:.2f}M, {:.2f}M FLOPs".format(params / 1e6, flops / 1e6))
