from time import time
from os import get_terminal_size as columns

__all__ = ["ProgressBar"]
__author__ = "YueLin"

# TODO: When running in PyCharm, os.get_terminal_size() will fail,
#  the following code is a temporary solution, but it's not elegant
try:
    columns()
except OSError:
    class Columns:
        def __init__(self):
            self.columns = 100
    columns = Columns


class ProgressBar:
    def __init__(self, name: str, unit: str, total: int):
        self.name = name
        self.unit = unit
        self.total = total
        self.current = self.time = 0

    @property
    def eta(self) -> str:
        eta = int(round(
            (time() - self.time) * (self.total - self.current) / self.current
        ))
        hours, minutes, seconds = eta // 3600, (eta // 60) % 60, eta % 60
        if hours >= 24:
            days, hours = divmod(hours, 24)
            return "{}day{} {:02d}:{:02d}:{:02d}".format(
                days, 's' if days > 1 else '', hours, minutes, seconds
            )
        return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

    @property
    def speed(self) -> float:
        return self.current / (time() - self.time)

    def bar(self, info) -> str:
        progress = self.current / self.total
        info = "{}: [\000] {:.2f}% {:.2f} {}/s {} eta {}".format(
            self.name, 100 * progress, self.speed, self.unit, info, self.eta
        )
        length = columns().columns - len(info) - 1
        progress = int(round(length * progress))
        return info.replace(
            '\000', '>'.join(('-' * progress, ' ' * (length - progress)))
        )

    def show(self, info: str = '') -> None:
        self.current += 1
        print(self.bar(info), end='\n' if self.current == self.total else '\r')

    def reset(self) -> None:
        self.current, self.time = 0, time()
