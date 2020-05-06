import abc
import numpy as np

import torch
from torch.utils.data import Dataset

from utils import prepare_mnist, get_transform, show_images


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


class CoLocalisationMNIST(Dataset):

    def __init__(self, digit, mode='train', transform='T', test_split=0.3):
        assert 0 <= digit <= 9
        
        super(CoLocalisationMNIST, self).__init__(mode, transform, test_split)

    def __getitem__(self, idx):
        return self.mnist[idx]


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = MNISTAddition(mode='train', test_split=0.3)
    dataloader = DataLoader(dataset, batch_size=36)

    for img, label in dataloader:
        print(img.size())
        show_images(img)
        break
