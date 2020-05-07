import abc
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from albumentations import Compose, BboxParams, PadIfNeeded, ShiftScaleRotate

from utils import prepare_mnist, get_transform, show_images

__all__ = [
    'DistortedMNIST',
    'MNISTAddition',
    'CoLocalisationMNIST'
]


class BaseMNIST(Dataset, abc.ABC):

    def __init__(self, mode='train', transform=None, test_split=0.3):
        assert mode in ['train', 'val', 'test']

        _train = (mode in ['train', 'val'])
        
        self.mode = mode
        self.mnist = prepare_mnist(train=_train,
                                   transform=get_transform(type=transform))

        index_list = list(range(len(self.mnist)))

        if _train:
            split = int(len(self.mnist) * (1 - test_split))

            if self.mode == 'train':
                self.index_list = index_list[:split]
            elif self.mode == 'val':
                self.index_list = index_list[split:]
        else:
            self.index_list = index_list

    def __len__(self):
        return len(self.index_list)

    @abc.abstractclassmethod
    def __getitem__(self, idx):
        pass


class DistortedMNIST(Dataset):

    def __init__(self, mode='train', transform='R', test_split=0.3):
        assert transform in ['R', 'RTS', 'P', 'E']

        super(DistortedMNIST, self).__init__(mode, transform, test_split)

    def __getitem__(self, idx):
        return self.mnist[idx]


class MNISTAddition(BaseMNIST):

    def __init__(self, mode='train', test_split=0.3):
        super(MNISTAddition, self).__init__(mode, 'RTS', test_split)

    def __getitem__(self, idx):
        _idx = np.random.choice(self.index_list)

        image_1, label_1 = self.mnist[idx]
        image_2, label_2 = self.mnist[_idx]

        image = torch.cat([image_1, image_2], dim=0)
        label = label_1 + label_2

        return image, label


class CoLocalisationMNIST(BaseMNIST):

    def __init__(self, mode='train', test_split=0.3):
        super(CoLocalisationMNIST, self).__init__(mode, None, test_split)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]

        annotations = {
            'image': image,
            'bboxes': [[0.0, 0.0, 28, 28]],  # [x_min, y_min, x_max, y_max]
            'category_id': [label]
        }

        transform = Compose([
            PadIfNeeded(min_height=84,
                        min_width=84,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=1.0),
            ShiftScaleRotate(shift_limit=1 / 3.,
                             scale_limit=0.0,
                             rotate_limit=0.0,
                             border_mode=cv2.BORDER_CONSTANT,
                             value=0,
                             interpolation=cv2.INTER_NEAREST,
                             p=1.0),
        ],
            bbox_params=BboxParams(format='pascal_voc',
                                   label_fields=['category_id'])
        )

        augmented = transform(**annotations)

        image = ToTensor()(augmented['image'])  # (1, 84, 84)
        bbox = torch.Tensor(augmented['bboxes'][0])  # (4,)

        return image, bbox, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = CoLocalisationMNIST(mode='train', test_split=0.3)
    dataloader = DataLoader(dataset, batch_size=4)

    for imgs, bboxes, labels in dataloader:
        print(imgs.size())
        print(bboxes.size())
        print(bboxes)
        print(labels.size())
        print(labels)
        show_images(imgs)
        break
