from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from cub_dataset import CUBDataset
import torch


class DRODataset(Dataset):
    def __init__(self,
                 dataset: CUBDataset,
                 n_groups: int,
                 n_classes: int,
                 group_str_fn=None):
        self.dataset = dataset
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        group_array = []
        y_array = []

        for _, y, g in self:
            group_array.append(g)
            y_array.append(y)

        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(
            1) == self._group_array).sum(dim=1).float()
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(
            1) == self._y_array).sum(dim=1).float()

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x, _, _ in self:
            print(x)
            return x.size()

    def get_loader(self, train, reweight_groups, **kwargs):
        if not train:
            shuffle = False
            sampler = None
        elif not reweight_groups:  # Training but not reweighting
            shuffle = True
            sampler = None
        else:
            # Training and reweighting
            # When the --robust flag is not set, reweighting changes the loss function
            # from the normal ERM (average loss over each training example)
            # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # When the --robust flag is set, reweighting does not change the loss function
            # since the minibatch is only used for mean gradient estimation for each group separately
            group_weights = len(self) / self._group_counts
            weights = group_weights[self._group_array]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(
                weights, len(self), replacement=True)
            shuffle = False
        loader = DataLoader(
            self,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs)
        return loader
