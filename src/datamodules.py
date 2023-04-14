import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.transforms.autoaugment import AutoAugment
import os

import torchvision.datasets as datasets


def get_imagenet(data_path, train, no_transform=False):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if train:
        transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomCrop(224, padding=4),
            AutoAugment(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if no_transform:
        transform = None

    data_path = os.path.join(data_path, 'train_fixed_size' if train else 'val_fixed_size')
    return datasets.ImageFolder(root=data_path, transform=transform)


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 batch_size: int = 512,
                 num_workers: int = 10,
                 prefetch_factor: int = 2,
                 sampling_policy: str = 'random'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.sampling_policy_name = sampling_policy
        self.sampling_policy = None

    def setup(self, stage=None):

        self.train_set = get_imagenet(data_path=self.data_dir, train=True)
        self.val_set = get_imagenet(data_path=self.data_dir, train=False)

    def train_dataloader(self):
        print(self.batch_size)
        print(self.num_workers)

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2  #todo check if this is best
        )

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True,
                          prefetch_factor=self.prefetch_factor)  # todo check if this is best
