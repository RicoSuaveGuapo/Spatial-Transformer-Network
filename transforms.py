import cv2
import numpy as np

from torchvision import transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2


# see https://hackmd.io/njZgW_rWQ6yyG0XL4wjI8g?view&fbclid=IwAR2ZzLbcxF66Kxc92UD7P8okslcBgYy_miYiYltrCXWje4t-hxGNBdzi7eA#Use-with-torchvisiontransforms
# for the reason of pre_transform and post_transform

def pil2array(image):
    image = np.array(image)

    if image.ndim != 3:
        image = np.expand_dims(image, axis=-1)

    return image


def project_transform(image):
    size = (28, 28)
    mean, std = 0.0, 5.0

    src = np.float32([[0, 0], [0, 28], [28, 0], [28, 28]])
    dst = src + np.random.randn(4, 2) * std + mean
    dst = dst.astype(np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(image, M, size)

    return image


def get_transforms(type: str):
    assert type in ['R', 'RTS', 'P', 'E', 'T', 'TU']

    if type in ['T', 'TU']:
        pre_transform = transforms.Lambda(lambd=pil2array)

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
                                  p=1.0),
        ],
            bbox_params=albu.BboxParams(format='pascal_voc',
                                        label_fields=['category_id'])
        )

        cluster_transform = None if type != 'TU' else albu.Compose([
            albu.RandomCrop(height=6, width=6),
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
            transforms.RandomRotation(degrees=90),  # if a bug appears at here
            # transforms.RandomRotation(degrees=90, fill=(0,)),  # use this function instead
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
                                    scale=(0.7, 1.2),  # random scaling
                                    fillcolor=0),
            transforms.ToTensor(),
        ])

        post_transform = None
    elif type == 'P':
        pre_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0.0,
                                    translate=None,
                                    scale=(0.75, 1.0),
                                    fillcolor=0),
            transforms.Lambda(lambd=pil2array),
            transforms.Lambda(lambd=project_transform),
            transforms.ToTensor()
        ])

        post_transform = None
    elif type == 'E':
        pre_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0.0,
                                    translate=None,
                                    scale=(0.75, 1.0),
                                    fillcolor=0),
            transforms.Lambda(lambd=pil2array)
        ])

        # TODO: need to tune the parameters
        post_transform = albu.Compose([
            albu.ElasticTransform(alpha=0.0,
                                  sigma=1.5,
                                  alpha_affine=3.0,
                                  border_mode=cv2.BORDER_CONSTANT,
                                  value=0,
                                  p=1.0),
            ToTensorV2()
        ])
        
    return pre_transform, post_transform, None
