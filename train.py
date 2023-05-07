import argparse as arg
from os.path import join

import torch

import data
import utils
import models
from val import Evaluator

__author__ = "YueLin"


class Trainer:
    def __init__(self,
                 model: models.Any,
                 dataset: str,
                 lr: float = 1e-3,
                 device: str = None,
                 pretrained: str = None,
                 l2regularization: float = 5e-5,
                 **kwargs):
        self.name = models.model_name(model)
        self.loss = utils.SoftCrossEntropyLoss()
        self.data = data.Dataset(dataset, "train", **kwargs)
        self.eval = Evaluator(data.Dataset(dataset, "val", **kwargs))
        self.device = torch.device("cuda" if device is None else device)
        self.model: models.Module = model(self.data.num_classes).to(self.device)
        if self.name in ("Swin Transformer", "ConvNext"):
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr, weight_decay=l2regularization
            )
        elif self.name.startswith("R"):
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr,
                momentum=0.9, weight_decay=l2regularization
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr, weight_decay=l2regularization
            )
        if pretrained:
            self.model.load_state_dict(torch.load(pretrained, self.device))

    def __str__(self):
        return "{} Trainer".format(self.name)

    def ward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> float:
        predicts = self.model(inputs.to(self.device))
        (loss := self.loss(predicts, outputs.to(self.device))).backward()
        return loss.item()

    def train(self,
              batch: int,
              epochs: int,
              save_path: str,
              warmup: int = 1,
              num_workers: int = 0) -> None:
        best, n = 0, 0
        self.model.train()
        steps = len(self.data) // batch
        dataset = data.DataLoader(
            self.data, batch, True, drop_last=True, num_workers=num_workers
        )
        progress = utils.ProgressBar("Train", "batches", steps)
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: step / (warmup * steps)
        ) if warmup else None
        if self.name == "ResNet":
            scheduler2 = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 1
            )
        else:
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, epochs
            )
        for epoch in range(epochs):
            loss = 0
            progress.reset()
            print(f"Epoch: {epoch + 1}/{epochs}")
            for item, batch in enumerate(dataset):
                self.optimizer.zero_grad()
                loss += self.ward(*batch)
                self.optimizer.step()
                if epoch < warmup:
                    scheduler1.step()
                progress.show("loss={:.2f}".format(loss / (item + 1)))
            if self.name != "ResNet":
                scheduler2.step()
            torch.save(self.model.state_dict(), join(save_path, "last.pt"))
            if (acc := self.eval(self.model)) > best:
                best, n = acc, 0
                torch.save(self.model.state_dict(), join(save_path, "best.pt"))
            else:
                n += 1
                if n == 10 and self.name == "ResNet":
                    n = 0
                    scheduler2.step()


class Distiller(Trainer):
    def __init__(self,
                 weights: str,
                 dataset: str,
                 teacher: models.Any,
                 student: models.Module,
                 lr: float = 1e-3,
                 alpha: float = 0.7,
                 device: str = None,
                 pretrained: str = None,
                 temperature: float = 1,
                 l2regularization: float = 5e-5,
                 **kwargs):
        super(Distiller, self).__init__(
            student, dataset, lr, device,
            pretrained, l2regularization, **kwargs
        )
        self.alpha = 1 - alpha
        self.temper = temperature
        self.teacher = teacher(self.data.num_classes).to(self.device)
        self.teacher.load_state_dict(torch.load(weights, self.device))
        self.teacher.eval()

    def ward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> float:
        inputs = inputs.to(self.device)
        predicts = self.model(inputs)
        targets = self.teacher(inputs).detach()
        loss1 = self.loss(predicts, outputs.to(self.device))
        loss2 = self.loss(predicts, utils.softmax(targets / self.temper))
        (loss := self.alpha * loss1 + (1 - self.alpha) * loss2).backward()
        return loss.item()


def main(args: arg.Namespace) -> None:
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = eval("models.{}".format(args.model))
    if rep := models.model_type(model) == "Rep":
        model = models.ConvNet
    if args.method == "train":
        trainer = Trainer(
            model,
            args.data,
            args.lr, device,
            args.pretrained,
            args.weight_decay,
            test_size=args.test_size,
            train_size=args.train_size,
            label_smooth=args.label_smooth
        )
    elif args.method == "distillation":
        teacher = eval("models.{}".format(args.teacher))
        if models.model_type(teacher) != "Base":
            raise ValueError("The teacher model must be the Base model")
        if (weights := args.teacher_weights) is None:
            raise ValueError("No teacher model weight file specified")
        trainer = Distiller(
            weights, args.data,
            teacher, model, args.lr,
            args.alpha, device, args.pretrained,
            args.temperature, args.weight_decay,
            test_size=args.test_size, train_size=args.train_size,
            label_smooth=args.label_smooth
        )
    else:
        raise ValueError("--method only supports train or distillation")
    trainer.train(
        args.batch_size, args.epochs, args.save, args.warmup, args.num_workers
    )
    if rep:
        pt = join(args.save, "best.pt")
        vgg = models.ConvNet(trainer.data.num_classes)
        vgg.load_state_dict(torch.load(pt, torch.device("cpu")))
        torch.save(
            models.RepVgg(trainer.data.num_classes, vgg).state_dict(), pt
        )


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument(
        "--save", help="Save path"
    )
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
        "--pretrained", default=None, help="Weights file"
    )
    parser.add_argument(
        "--teacher", default=None, help="Teacher model name"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size"
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Training epochs"
    )
    parser.add_argument(
        "--method", default="train", help="train or distillation"
    )
    parser.add_argument(
        "--warmup", default=1, type=int, help="Linear warmup epochs"
    )
    parser.add_argument(
        "--alpha", default=0.7, type=float, help="Distillation alpha"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of threads"
    )
    parser.add_argument(
        "--label_smooth", default=0.1, type=float, help="Label smooth"
    )
    parser.add_argument(
        "--test_size", default=160, type=int, help="Image size when test"
    )
    parser.add_argument(
        "--train_size", default=128, type=int, help="Image size when train"
    )
    parser.add_argument(
        "--teacher_weights", default=None, help="Teacher model weights file"
    )
    parser.add_argument(
        "--weight_decay", default=5e-5, type=float, help="L2 regularization"
    )
    parser.add_argument(
        "--temperature", default=1, type=float, help="Distillation temperature"
    )
    main(parser.parse_args())
