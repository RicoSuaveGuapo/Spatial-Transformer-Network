import abc
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import prepare_mnist, show_images
from transforms import get_transforms


class BaseMNIST(Dataset, abc.ABC):

    def __init__(self, mode='train', transform_type=None, val_split=0.3, seed=42):
        """Base class for three MNIST tasks

        Keyword Arguments:
            mode {str} -- initialize dataset for training, validation, or test (default: {'train'})
            transform_type {str} -- one of 'R', 'RTS', 'P', 'E', 'T', or 'TU' (default: {None})
            val_split {float} -- ratio for validation (default: {0.3})
            seed {int} -- seed for generating the same training or validation dataset (default: {42})
        """

        assert mode in ['train', 'val', 'test']
        super(BaseMNIST, self).__init__()

        train = mode in ['train', 'val']

        self.mode = mode
        self.pre_transform, self.post_transform, self.cluster_transform = get_transforms(type=transform_type)
        self.mnist = prepare_mnist(train=train, transform=self.pre_transform)

        np.random.seed(seed)
        index_list = list(range(len(self.mnist)))
        index_list = np.random.permutation(index_list)
        
        if train:
            split = int(len(self.mnist) * (1 - val_split))

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


class DistortedMNIST(BaseMNIST):

    def __init__(self, mode='train', transform_type='R', val_split=0.3, seed=42):
        assert transform_type in ['R', 'RTS', 'P', 'E']
        super(DistortedMNIST, self).__init__(mode, transform_type, val_split, seed)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]

        if self.post_transform:
            augmented = self.post_transform(image=image)

            if isinstance(augmented, dict):
                # transformation of albumentations
                image = augmented['image']
            else:
                # custom transformation
                image = augmented

        return image, label


class MNISTAddition(BaseMNIST):

    def __init__(self, mode='train', val_split=0.3, seed=42):
        super(MNISTAddition, self).__init__(mode, 'RTS', val_split, seed)

    def __getitem__(self, idx):
        random_idx = np.random.choice(self.index_list)

        image_1, label_1 = self.mnist[idx]
        image_2, label_2 = self.mnist[random_idx]

        image = torch.cat([image_1, image_2], dim=0)  # (2, 42, 42)
        label = label_1 + label_2

        return image, label


class CoLocalisationMNIST(BaseMNIST):

    def __init__(self, mode='train', transform_type='T', val_split=0.3, seed=42):
        assert transform_type in ['T', 'TU']
        super(CoLocalisationMNIST, self).__init__(mode, transform_type, val_split, seed)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]

        bboxes = [[0.0, 0.0, 28, 28]]  # [x_min, y_min, x_max, y_max]
        category_id = [label]
        augmented = self.post_transform(image=image, bboxes=bboxes, category_id=category_id)
        image = augmented['image']
        bbox = augmented['bboxes'][0]

        if self.cluster_transform:
            random_idx = np.random.choice(self.index_list)
            random_image, _ = self.mnist[random_idx]
            
            for _ in range(16):
                augmented = self.cluster_transform(image=random_image)
                random_crop = augmented['image']
                image += random_crop

        image[image > 255] = 255
        image = transforms.ToTensor()(image)  # (1, 84, 84)
        bbox = torch.tensor(bbox)  # (4,)  # [x_min, y_min, x_max, y_max]

        return image, bbox, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # test whether splitting is correct or not
    seed = 42
    train_dataset = CoLocalisationMNIST(mode='train', transform_type='T', val_split=0.3, seed=seed)
    val_dataset = CoLocalisationMNIST(mode='val', transform_type='T', val_split=0.3, seed=seed)
    
    overlap_index = set(train_dataset.index_list) & set(val_dataset.index_list)

    if overlap_index:
        print('overlap index:', overlap_index)
        raise ValueError('Some wrong when splitting training and validation datasets')
    else:
        print('Splitting training and validation datasets successfully')

    # try to show transformed images
    dataset = DistortedMNIST(mode='train', transform_type='E', val_split=0.3)
    dataloader = DataLoader(dataset, batch_size=64)

    for imgs, labels in dataloader:
        print('images:', imgs.size())
        print('labels:', labels.size())
        print(labels)
        show_images(imgs)
        break

    dataset = MNISTAddition(mode='train', val_split=0.3)
    dataloader = DataLoader(dataset, batch_size=64)

    for imgs, labels in dataloader:
        print('images:', imgs.size())
        print('labels:', labels.size())
        print(labels)
        show_images(imgs)
        break
    
    dataset = CoLocalisationMNIST(mode='train', transform_type='TU', val_split=0.3)
    dataloader = DataLoader(dataset, batch_size=64)

    for imgs, bboxes, labels in dataloader:
        print('images:', imgs.size())
        print('bboxes:', bboxes.size())
        print(bboxes)
        print('labels:', labels.size())
        print(labels)
        show_images(imgs)
        break
