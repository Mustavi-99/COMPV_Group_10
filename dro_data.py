from cub_dataset import CUBDataset
from dro_dataset import DRODataset


def get_confounder_splits(data_dir: str, augment_data=False):
    full_dataset = CUBDataset(data_dir=data_dir,
                              augment_data=augment_data)
    # change this list later
    splits = ['train', 'val', 'test']
    subsets = full_dataset.get_splits(splits=splits)
    dro_subsets = []
    for split in splits:
        dro_dataset = DRODataset(dataset=subsets[split],
                                 n_classes=full_dataset.n_classes,
                                 n_groups=full_dataset.n_groups)
        dro_subsets.append(dro_dataset)

    return dro_subsets
