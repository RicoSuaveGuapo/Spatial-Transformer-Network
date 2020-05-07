import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.datasets import MNIST

import albumentations as albu


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
    plt.figure(figsize=(8, 8))

    for i, img in enumerate(imgs):
        plt.subplot(nrows, ncols, i + 1)
        plt.axis('off')
        plt.imshow(img, cmap='gray')

    plt.show()


def get_transforms(type: str):
    assert type in ['R', 'RTS', 'P', 'E', 'T', 'TU', None]

    if type in ['T', 'TU']:
        pre_transform = transforms.Lambda(lambda img: np.array(img))

        post_transform = albu.Compose([
            albu.PadIfNeeded(min_height=84,
                             min_width=84,
                             border_mode=cv2.BORDER_CONSTANT,
                             value=0,
                             p=1.0),
            albu.ShiftScaleRotate(shift_limit=(84 - 28) / (84 * 2),
                                  scale_limit=0.0,
                                  rotate_limit=0.0,
                                  border_mode=cv2.BORDER_CONSTANT,
                                  value=0,
                                  interpolation=cv2.INTER_NEAREST,
                                  p=1.0),
        ],
            bbox_params=albu.BboxParams(format='pascal_voc',
                                        label_fields=['category_id'])
        )

        cluster_transform = None if type != 'TU' else albu.Compose([
            albu.RandomCrop(height=6, width=6, ),
            albu.PadIfNeeded(min_height=84,
                             min_width=84,
                             border_mode=cv2.BORDER_CONSTANT,
                             value=0,
                             p=1.0),
            albu.ShiftScaleRotate(shift_limit=(84 - 6) / (84 * 2),
                                  scale_limit=0.0,
                                  rotate_limit=0.0,
                                  border_mode=cv2.BORDER_CONSTANT,
                                  value=0,
                                  p=1.0)
        ])
        
        return pre_transform, post_transform, cluster_transform

    if type == 'R':
        pre_transform = transforms.Compose([
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor()
        ])

        post_transform = None
    elif type == 'RTS':
        padding = (42 - 28) // 2
        shift = (42 - 28) / (42 * 2)

        pre_transform = transforms.Compose([
            transforms.Pad(padding=padding),
            transforms.RandomAffine(degrees=45,  # randomly rotate
                                    translate=(shift, shift),  # randomly place
                                    scale=(0.7, 1.2),
                                    fillcolor=0),  # random scaling
            transforms.ToTensor(),
        ])

        post_transform = None
    elif type == 'P':
        # TODO:
        pass
    elif type == 'E':
        pre_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0.0,
                                    translate=None,
                                    scale=(1.0, 1.0),
                                    fillcolor=0),
            transforms.Lambda(lambda img: np.array(img))
        ])

        # TODO: need to tune the parameters
        post_transform = albu.ElasticTransform(alpha=0.0,
                                               sigma=1.5,
                                               alpha_affine=3.0,
                                               border_mode=cv2.BORDER_CONSTANT,
                                               value=0,
                                               p=1.0)
    
    return pre_transform, post_transform, None


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
