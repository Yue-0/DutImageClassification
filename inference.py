import torch

import data
import models as zoo
from utils import ProgressBar

__author__ = "YueLin"


class Predictor:
    def __init__(self,
                 dataset: str,
                 models: list[zoo.Any],
                 parameters: list[str],
                 device: str = None,
                 **kwargs):
        self.models = []
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        assert (n := len(models)) == len(parameters)
        self.dataset = data.Dataset(dataset, "test", **kwargs)
        for m in range(n):
            model = models[m](self.dataset.num_classes).to(self.device)
            model.load_state_dict(torch.load(parameters[m], self.device))
            self.models.append(model)
            model.eval()
        self.softmax = torch.nn.Softmax(0)
        self.progress = ProgressBar("Predict", "images", len(self.dataset))

    def predict(self, output: str, score: bool = False) -> None:
        self.progress.reset()
        csv = open(output, "w")
        if score:
            csv.write("Id")
            for c in range(self.dataset.num_classes):
                csv.write(",score_{}".format(c))
            csv.write('\n')
        else:
            csv.write("Id,Prediction\n")
        with torch.no_grad():
            for image, name in self.dataset:
                result = sum([
                    model(image.to(self.device)).sum(0) for model in self.models
                ]).cpu()
                if score:
                    csv.write(name)
                    result = self.softmax(result)
                    for p in result / torch.sum(result):
                        csv.write(",{}".format(p.item()))
                    csv.write('\n')
                else:
                    result = torch.argmax(result).item()
                    csv.write("{},{}\n".format(name, result))
                self.progress.show()
        csv.close()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--data", help="Data path"
    )
    parser.add_argument(
        "--models", nargs='+', help="Model name"
    )
    parser.add_argument(
        "--weights", nargs='+', help="Weights file"
    )
    parser.add_argument(
        "--device", default=None, help="Device name"
    )
    parser.add_argument(
        "--output", type=str, help="Output csv file name"
    )
    parser.add_argument(
        "--test_size", default=160, type=int, help="Image size when test"
    )
    parser.add_argument(
        "--train_size", default=128, type=int, help="Image size when train"
    )
    parser.add_argument(
        "-s", "--score", action="store_true", default=False, help="Output score"
    )
    parser = parser.parse_args()

    names = parser.models
    for i in range(len(names)):
        names[i] = eval("zoo.{}".format(names[i]))

    Predictor(
        parser.data, names, parser.weights, parser.device,
        test_size=parser.test_size, train_size=parser.train_size
    ).predict(parser.output, parser.score)
