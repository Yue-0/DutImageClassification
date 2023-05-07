import torch

__all__ = ["Softmax", "SoftCrossEntropyLoss"]
__author__ = "YueLin"


class Softmax(torch.nn.Softmax):
    def __init__(self):
        super(Softmax, self).__init__(1)


class Log2Softmax(torch.nn.Module):
    def __init__(self):
        super(Log2Softmax, self).__init__()
        self.softmax = Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.log2(self.softmax(x))


# class KLDivLoss(torch.nn.Module):
#     def __init__(self):
#         super(KLDivLoss, self).__init__()
#         self.log2softmax = Log2Softmax()
#         self.kl = torch.nn.KLDivLoss(reduction="batch""mean", log_target=True)
#
#     def forward(self, *inputs: tuple[torch.Tensor]) -> torch.Tensor:
#         return self.kl(*map(self.log2softmax, inputs))


class SoftCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()
        self.log2softmax = Log2Softmax()

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.sum(q * self.log2softmax(p)) / q.shape[0]
