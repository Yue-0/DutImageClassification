from torchvision.models import swin_transformer

__all__ = ["SwinTransformerN", "SwinTransformerT"]
__author__ = "YueLin"


class SwinTransformerN(swin_transformer.SwinTransformer):
    def __init__(self, num_classes: int):
        super(SwinTransformerN, self).__init__(
            embed_dim=64,
            patch_size=[4, 4],
            window_size=[8, 8],
            depths=[2, 2, 4, 2],
            num_heads=[2, 4, 8, 16],
            num_classes=num_classes,
            stochastic_depth_prob=0.1,
            block=swin_transformer.SwinTransformerBlockV2,
            downsample_layer=swin_transformer.PatchMergingV2
        )


def swin_transformer_t(num_classes: int) -> swin_transformer.SwinTransformer:
    st = swin_transformer.swin_v2_t(
        weights=swin_transformer.Swin_V2_T_Weights.DEFAULT
    )
    st.head = swin_transformer.nn.Linear(st.head.in_features, num_classes)
    return st


SwinTransformerT = swin_transformer_t


if __name__ == "__main__":
    from torch import randn
    from thop import profile
    model = SwinTransformerN(50)
    inputs = randn(1, 3, 128, 128)
    flops, params = profile(model, (inputs,))
    print("{:.2f}M, {:.2f}M FLOPs".format(params / 1e6, flops / 1e6))
