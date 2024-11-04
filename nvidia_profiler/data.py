import torchvision

from torch.utils.data import random_split
from torchvision.datasets import VisionDataset


__all__ = [
    'getDatasets'
]


def getDatasets(test_ratio: float = 0.33) -> (VisionDataset, VisionDataset):
    # TODO transform
    dataset: VisionDataset = torchvision.datasets.OxfordIIITPet(root='data', download=True, transform=None)

    # split into training and test dataset
    num_images: int = len(dataset)
    num_images_test: int = int(test_ratio * num_images)
    num_images_train: int = num_images - num_images_test

    train_dataset, test_dataset = random_split(dataset, [num_images_train, num_images_test])
    return train_dataset, test_dataset
