from torch.utils.data import DataLoader
from monai.data import Dataset
import pickle

from .transforms import (
    train_transforms,
    public_transforms,
    valid_transforms,
    tuning_transforms,
    unlabeled_transforms,
)
from .utils import split_train_valid, path_decoder

DATA_LABEL_DICT_PICKLE_FILE = "./train_tools/data_utils/custom/modalities.pkl"

__all__ = [
    "get_dataloaders_labeled",
    "get_dataloaders_public",
    "get_dataloaders_unlabeled",
]


def get_dataloaders_labeled(
    root,
    mapping_file,
    mapping_file_tuning=False,
    join_mapping_file=None,
    valid_portion=0.0,
    batch_size=8,
    amplified=False,
    relabel=False,
):
    """Set DataLoaders for labeled datasets.

    Args:
        root (str): root directory
        mapping_file (str): json file for mapping dataset
        mapping_file_tuning (str, optional): json file for mapping tuning dataset. Defaults to None.
        join_mapping_file (str, optional):
        valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 8.
        amplified (bool, optional):
        relabel (bool, optional):
    Returns:
        dict: dictionary of data loaders.
    """
    # TODO: Add missing docstring
    # Get list of data dictionaries from decoded paths
    data_dicts = path_decoder(root, mapping_file)


    if amplified:
        with open(DATA_LABEL_DICT_PICKLE_FILE, "rb") as f:
            data_label_dict = pickle.load(f)

        data_point_dict = {}

        for label, data_lst in data_label_dict.items():
            data_point_dict[label] = []

            for d_idx in data_lst:
                try:
                    data_point_dict[label].append(data_dicts[d_idx])
                except IndexError:  # TODO: Does it throw IndexError?
                    print(label, d_idx)

        data_dicts = []

        for label, data_points in data_point_dict.items():
            len_data_points = len(data_points)

            if len_data_points >= 50:
                data_dicts += data_points
            else:
                for i in range(50):
                    data_dicts.append(data_points[i % len_data_points])

    data_transforms = train_transforms

    if join_mapping_file is not None:
        data_dicts += path_decoder(root, join_mapping_file)
        data_transforms = public_transforms

    if relabel:
        for elem in data_dicts:
            cell_idx = int(elem["label"].split("_label.tiff")[0].split("_")[-1])
            if cell_idx in range(340, 499):
                new_label = elem["label"].replace(
                    "/data/CellSeg/Official/Train_Labeled/labels/",
                    "/CellSeg/pretrained_train_ext/",
                )
                elem["label"] = new_label
    print("!")
    # Split datasets as Train/Valid
    train_dicts, valid_dicts = split_train_valid(
        data_dicts, valid_portion=valid_portion
    )

    # Obtain datasets with transforms
    train_set = Dataset(train_dicts, transform=data_transforms)
    valid_set = Dataset(valid_dicts, transform=valid_transforms)

    # Set dataloader for train_set
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=5
    )

    # Set dataloader for valid_set (Batch size is fixed as 1)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False,)

    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
    }

    if mapping_file_tuning:
        tuning_dicts = path_decoder(root, mapping_file_tuning, no_label=True)
        tuningset = Dataset(tuning_dicts, transform=tuning_transforms)
        tuning_loader = DataLoader(tuningset, batch_size=1, shuffle=False)
        dataloaders["tuning"] = tuning_loader

    return dataloaders


def get_dataloaders_public(
    root, mapping_file, valid_portion=0.0, batch_size=8,
):
    """Set DataLoaders for labeled datasets.

    Args:
        root (str): root directory
        mapping_file (str): json file for mapping dataset
        valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 8.
    Returns:
        dict: dictionary of data loaders.
    """

    # Get list of data dictionaries from decoded paths
    data_dicts = path_decoder(root, mapping_file)

    # Split datasets as Train/Valid
    train_dicts, _ = split_train_valid(data_dicts, valid_portion=valid_portion)

    train_set = Dataset(train_dicts, transform=public_transforms)
    # Set dataloader for train_set
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=5
    )

    # Form dataloaders as dictionary
    dataloaders = {
        "public": train_loader,
    }

    return dataloaders


def get_dataloaders_unlabeled(
    root, mapping_file, batch_size=8, shuffle=True, num_workers=5,
):
    """Set dataloaders for unlabeled dataset."""
    # Get list of data dictionaries from decoded paths
    unlabeled_dicts = path_decoder(root, mapping_file, no_label=True, unlabeled=True)

    # Obtain datasets with transforms
    unlabeled_dicts, _ = split_train_valid(unlabeled_dicts, valid_portion=0)
    unlabeled_set = Dataset(unlabeled_dicts, transform=unlabeled_transforms)

    # Set dataloader for Unlabeled dataset
    unlabeled_loader = DataLoader(
        unlabeled_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    dataloaders = {
        "unlabeled": unlabeled_loader,
    }

    return dataloaders


def get_dataloaders_unlabeled_pseudo(
    root, mapping_file, batch_size=8, shuffle=True, num_workers=5,
):

    # Get list of data dictionaries from decoded paths
    unlabeled_pseudo_dicts = path_decoder(
        root, mapping_file, no_label=False, unlabeled=True
    )

    # Obtain datasets with transforms
    unlabeled_pseudo_dicts, _ = split_train_valid(
        unlabeled_pseudo_dicts, valid_portion=0
    )
    unlabeled_pseudo_set = Dataset(unlabeled_pseudo_dicts, transform=train_transforms)

    # Set dataloader for Unlabeled dataset
    unlabeled_pseudo_loader = DataLoader(
        unlabeled_pseudo_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    dataloaders = {"unlabeled": unlabeled_pseudo_loader}

    return dataloaders
