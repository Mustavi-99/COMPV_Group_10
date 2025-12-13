# from confounder_dataset import ConfounderDataset
import os
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, Subset
import numpy as np


class CUBDataset(Dataset):
    def __init__(self,
                 data_dir,
                 target_name=None,
                 confounder_names=None,
                 augment_data=False):

        self.data_dir = data_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1

        self.n_groups = 4
        self.group_array = (self.y_array * (self.n_groups / 2) +
                            self.confounder_array).astype('int')

        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.train_transform = get_transform_cub(
            train=True,
            augment_data=augment_data)

        self.eval_transform = get_transform_cub(
            train=False,
            augment_data=augment_data)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        img_filename = os.path.join(self.data_dir, self.filename_array[idx])
        img = Image.open(img_filename).convert("RGB")

        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and self.eval_transform:
            img = self.eval_transform(img)

        return img, y, g

    def get_splits(self, splits: list):
        subsets = {}
        for split in splits:
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            subsets[split] = Subset(self, indices)
        return subsets


def get_transform_cub(train, augment_data):
    scale = 256.0 / 224.0
    target_resolution = (224, 224)  # for resnet

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize(
                (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform
