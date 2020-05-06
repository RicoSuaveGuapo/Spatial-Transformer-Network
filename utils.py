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
    assert type in ['R', 'RTS', 'P', 'E', 'T', 'TU']

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
    elif type == 'T':
        padding = (84 - 28) // 2
        shift = (84 - 28) / (84 * 2)

        transform = transforms.Compose([
            transforms.Pad(padding=padding),
            transforms.RandomAffine(degrees=0,
                                    translate=(shift, shift))
        ])
    else:
        # TODO:
        pass

    return transform
