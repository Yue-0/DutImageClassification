import os
from sys import argv
from glob import glob

import cv2
import numpy as np
from paddlehub import Module

__author__ = "YueLin"

PATH = argv[1]

T7 = "starry_night.t7"
http = "http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/"
os.system("wget {}{}".format(http, T7))
del http

VAN = cv2.dnn.readNetFromTorch(T7)
VAN.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

ANIME = Module(name="anime""gan_v2_ha""yao_99")

N = len(glob(os.path.join(PATH, "val", '*', "*.jpg")))
N += len(glob(os.path.join(PATH, "train", '*', "*.jpg")))
N *= 3


def van(image: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    VAN.setInput(cv2.dnn.blobFromImage(
        image, 2, (w, h), (127.5,) * 3, swapRB=False, crop=False
    ))
    image = VAN.forward()
    image = image.reshape(3, *image.shape[2:])
    for channel in range(3):
        image[channel] += 127.5
    return image.transpose(1, 2, 0)


def anime(image: np.ndarray) -> np.ndarray:
    return ANIME.style_transfer([image])[0]


def sketch(image: np.ndarray, p: float = 0.2) -> np.ndarray:
    dx, dy = map(lambda g: p * g, np.gradient(image)[:2])
    image = (255 / np.sqrt(1 + dx ** 2 + dy ** 2)).clip(0, 255)
    image = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


n = 0
for style in ("anime", "sketch", "van"):
    convert = eval(style)
    os.mkdir(os.path.join(PATH, style))
    for mode in ("train", "val"):
        os.mkdir(path := os.path.join(PATH, style, mode))
        for cls in os.listdir(os.path.join(PATH, mode)):
            os.mkdir(os.path.join(path, cls))
            # TODO: Multiple images can be converted as a mini-batch
            for jpg in glob(os.path.join(PATH, mode, cls, "*.jpg")):
                n += 1
                img = convert(cv2.imread(jpg))
                cv2.imwrite(os.path.join(path, cls, os.path.split(jpg)[1]), img)
                print("Complete: {}, remaining: {}, progress: {:.2f}%{}".format(
                    n, N - n, 100 * n / N, ' ' * 20
                ), end='\r')
os.system("rm {}".format(T7))
print()
