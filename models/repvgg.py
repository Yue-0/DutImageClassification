import torch

__all__ = ["ConvNet", "RepVgg"]
__author__ = "YueLin"

nn = torch.nn


class RepVggBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(RepVggBlock, self).__init__()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ident = in_channels == out_channels
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        if not self.ident:
            self.bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn1(self.conv1x1(x)) + self.bn2(self.conv3x3(x))
        if self.ident:
            y += self.bn(x)
        return y


class ConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super(ConvNet, self).__init__()
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(2)
        self.conv1 = RepVggBlock(3, 64)
        self.conv2 = RepVggBlock(64, 64)
        self.conv3 = RepVggBlock(64, 128)
        self.conv4 = RepVggBlock(128, 128)
        self.conv5 = RepVggBlock(128, 256)
        self.conv6 = RepVggBlock(256, 256)
        self.conv7 = RepVggBlock(256, 256)
        self.conv8 = RepVggBlock(256, 256)
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(4096, 1024), self.relu,
            nn.Linear(1024, 1024), self.relu,
            nn.Linear(1024, num_classes)
        )

    def __len__(self):
        layers = 1
        while hasattr(self, f"conv{layers}"):
            layers += 1
        return layers - 1

    def __getitem__(self, item: int) -> RepVggBlock:
        return eval("self.conv{}".format(item))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv4(self.relu(self.conv3(x)))))
        x = self.pool(self.relu(self.conv6(self.relu(self.conv5(x)))))
        x = self.pool(self.relu(self.conv8(self.relu(self.conv7(x)))))
        return self.linear(x)


class RepVgg(ConvNet):
    def __init__(self, num_classes: int, cnn: ConvNet = None):
        super(RepVgg, self).__init__(num_classes)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 1)
        if cnn is not None:
            cnn.eval()
            for layer in range(1, 1 + len(self)):
                self.rep(cnn[layer], layer)
            self.linear.load_state_dict(cnn.linear.state_dict())
        self.eval()

    def rep(self, block: RepVggBlock, layer: int) -> None:
        k1 = block.bn1.weight / (block.bn1.running_var + block.bn1.eps) ** 0.5
        k2 = block.bn2.weight / (block.bn2.running_var + block.bn2.eps) ** 0.5
        b1 = block.bn1.bias - k1 * block.bn1.running_mean
        b2 = block.bn2.bias - k2 * block.bn2.running_mean
        w0, w2 = block.conv1x1.weight.data, block.conv3x3.weight.data
        w1 = torch.zeros_like(w2)
        w2 *= k2.expand(w2.shape[::-1]).permute(3, 2, 1, 0)
        w1[:, :, 1:2, 1:2] = w0 * k1.expand(w0.shape[::-1]).permute(3, 2, 1, 0)
        weight = w1 + w2
        conv = self[layer]
        conv.bias.data = b1 + b2
        conv.weight.data = weight
        if block.ident:
            w = torch.zeros_like(w2)
            e = torch.eye(w.shape[0]).resize(w.shape[0], w.shape[0], 1, 1)
            k = block.bn.weight / (block.bn.running_var + block.bn.eps) ** 0.5
            w[:, :, 1:2, 1:2] = e * k.expand(w0.shape[::-1]).permute(3, 2, 1, 0)
            conv.bias.data += block.bn.bias - k * block.bn.running_mean
            conv.weight.data += w


if __name__ == "__main__":
    from thop import profile

    inputs = torch.randn(1, 3, 128, 128)

    model1 = ConvNet(50)
    model1(inputs)
    model2 = RepVgg(50, model1)
    # x1, x2 = model1(inputs), model2(inputs)
    # print((x1 - x2).mean().item())

    flops, params = profile(model2, (inputs,))
    print("{:.2f}M, {:.2f}M FLOPs".format(params / 1e6, flops / 1e6))
