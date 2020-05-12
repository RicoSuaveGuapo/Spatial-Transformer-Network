import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.datasets import MNIST
from torchvision.utils import make_grid


def prepare_mnist(train, transform):
    dataset = MNIST(root='../data',
                    train=train,
                    download=True,
                    transform=transform)

    return dataset


def show_images(images: torch.Tensor):
    assert images.ndim == 4, 'only accept batch images'

    bz, c, h, w = images.size()

    images = images.view(bz * c, 1, h, w)
    images = make_grid(images, pad_value=255)
    images = images.numpy()
    images = np.transpose(images, (1, 2, 0))
    plt.imshow(images, cmap='gray')
    plt.axis('off')
    plt.show()


def compute_iou(bbox_1, bbox_2):
    x_min_1, y_min_1, x_max_1, y_max_1 = bbox_1
    x_min_2, y_min_2, x_max_2, y_max_2 = bbox_2

    assert x_min_1 < x_max_1
    assert y_min_1 < y_max_1
    assert x_min_2 < x_max_2
    assert y_min_2 < y_max_2

    x_min = max(x_min_1, x_min_2)
    y_min = max(y_min_1, y_min_2)
    x_max = min(x_max_1, x_max_2)
    y_max = min(y_max_1, y_max_2)

    overlap = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    area_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
    area_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2)

    iou = overlap / (area_1 + area_2 - overlap)

    return iou


if __name__ == "__main__":
    bbox_1 = 0, 0, 10, 10
    bbox_2 = 5, 5, 15, 15

    # overlap = 25
    # area_1 = area_2 = 100

    if compute_iou(bbox_1, bbox_2) == 25 / (100 + 100 - 25):
        print('testing `compute_iou` successfully')
