import torch

from data import Dataset
from utils import ProgressBar

__all__ = ["Evaluator"]
__author__ = "YueLin"


class Evaluator:
    def __init__(self, dataset: Dataset):
        self.data = dataset
        self.device = None
        self.progress = ProgressBar(" Val ", "images", len(dataset))

    def __call__(self, *args, **kwargs) -> float:
        return self.eval(*args, **kwargs)

    def eval(self, model: torch.nn.Module) -> float:
        model.eval()
        acc, n = 0, 0
        self.progress.reset()
        self.device = next(model.parameters()).device
        for images, label in self.data:
            n += (b := images.shape[0] // 10)
            predicts = model(images.to(self.device)).reshape(b, 10, -1)
            acc += torch.argmax(predicts.sum(1), 1).tolist().count(label)
            self.progress.show("acc={:.1f}%".format(100 * acc / n))
        model.train()
        return acc / n


if __name__ == "__main__":

    from argparse import ArgumentParser

    import torch
    from thop import profile

    import models

    parser = ArgumentParser()
    parser.add_argument(
        "--data", help="Data path"
    )
    parser.add_argument(
        "--model", help="Model name"
    )
    parser.add_argument(
        "--device", default=None, help="Device name"
    )
    parser.add_argument(
        "--weights", default=None, help="Weights file"
    )
    parser.add_argument(
        "--test_size", default=160, type=int, help="Image size when test"
    )
    parser.add_argument(
        "--train_size", default=128, type=int, help="Image size when train"
    )
    parser = parser.parse_args()

    device = parser.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data, device = Dataset(
        parser.data, "val", parser.test_size, parser.train_size
    ), torch.device(device)

    test = (torch.randn(1, 3, parser.train_size, parser.train_size).to(device),)

    name = eval("models.{}".format(parser.model))
    if name is models.ConvNet:
        raise TypeError("Please use RepVgg, not ConvNet")

    nn = name(data.num_classes).to(device)
    if parser.weights is None:
        raise RuntimeError("No weight file specified")
    nn.load_state_dict(torch.load(parser.weights, device))

    print("Accuracy: {:.2f}%".format(100 * Evaluator(data).eval(nn)))
    print("Parameters: {:.2f}M, FLOPs: {:.2f}M".format(
        *map(lambda n: n / 1e6, profile(nn, test)[::-1])
    ))
