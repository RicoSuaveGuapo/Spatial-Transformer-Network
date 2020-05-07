import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.datasets import MNIST


def prepare_mnist(train, transform):
    dataset = MNIST(root='../data',
                    train=train,
                    download=True,
                    transform=transform)

    return dataset


def show_images(images: torch.Tensor):
    assert images.ndim == 4

    c = images.size(1)

    if c == 1:
        imgs = [transforms.ToPILImage()(img) for img in images]
    if c == 2:
        imgs = []
        for img_pair in images:
            imgs += [transforms.ToPILImage()(img) for img in img_pair]

    n_img = len(imgs)
    ncols = 8
    nrows = np.ceil(n_img / ncols)

    for i, img in enumerate(imgs):
        plt.subplot(nrows, ncols, i + 1)
        plt.axis('off')
        plt.imshow(img)

    plt.show()


def get_transform(type: str):
    assert type in ['R', 'RTS', 'P', 'E', None]

    if type == 'R':
        transform = transforms.RandomRotation(90)
    elif type == 'RTS':
        padding = (42 - 28) // 2
        shift = (42 - 28) / (42 * 2)

        transform = transforms.Compose([
            transforms.Pad(padding=padding),
            transforms.RandomAffine(degrees=45,  # randomly rotate
                                    translate=(shift, shift),  # randomly place
                                    scale=(0.7, 1.2)),  # random scaling
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.1307,),
            #                      std=(0.3081,))
        ])
    elif type == 'P':
        # TODO:
        pass
    elif type == 'E':
        # TODO:
        pass
    else:
        # for transform (T) and (TU)
        transform = transforms.Lambda(lambda img: np.array(img))

    return transform


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
