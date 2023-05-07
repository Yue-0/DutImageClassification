from torchvision.models import convnext

__all__ = ["ConvNextTiny", "ConvNextNano"]
__author__ = "YueLin"


class ConvNextNano(convnext.ConvNeXt):
    """7.43M params, 576.71M FLOPs"""
    def __init__(self, num_classes: int):
        blocks = [
            convnext.CNBlockConfig(64, 128, 3),
            convnext.CNBlockConfig(128, 256, 3),
            convnext.CNBlockConfig(256, 256, 9),
            convnext.CNBlockConfig(256, None, 3)
        ]
        super(ConvNextNano, self).__init__(
            blocks, 0.1, num_classes=num_classes
        )


def conv_next_tiny(num_classes: int) -> convnext.ConvNeXt:
    cnn = convnext.convnext_tiny(
        weights=convnext.ConvNeXt_Tiny_Weights.DEFAULT
    )
    cnn.classifier.append(convnext.nn.Linear(
        cnn.classifier.pop(2).in_features, num_classes
    ))
    return cnn


ConvNextTiny = conv_next_tiny


if __name__ == "__main__":
    from torch import randn
    from thop import profile
    model = ConvNextNano(50)
    inputs = randn(1, 3, 128, 128)
    flops, params = profile(model, (inputs,))
    print("{:.2f}M, {:.2f}M FLOPs".format(params / 1e6, flops / 1e6))
