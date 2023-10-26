import os

from torch.utils.data import DataLoader
from monai.data import Dataset
import pickle

from .transforms import (
    TrainTransforms,
    ValidationTransforms,
)

from .utils import split_train_valid, path_decoder


__all__ = [
    "get_dataloaders",
]


def get_dataloaders(
    root,
    mapping_file,
    valid_portion=0.0,
    batch_size=8,
    amplified=False,
    modalities=None,
) -> dict:
    """Set DataLoaders for labeled datasets.
    Args:
        root (str): root directory
        mapping_file (str): json file for mapping dataset
        valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 8.
        amplified (bool, optional): if True, amplify the dataset. Defaults to False.
        modalities (str, optional): if amplified is True, path to modalities should be given. Defaults to None.
    Returns:
        dict: dictionary of data loaders.
    Raises:
        ValueError: if modalities is None.
        FileNotFoundError: if file with modalities does not exist.
    """

    data_dicts = path_decoder(root, mapping_file)

    if amplified:
        if modalities is None:
            raise ValueError("Path to modalities should be given.")
        if not os.path.isfile(modalities):
            raise FileNotFoundError(f"{modalities} does not exist.")

        with open(modalities, "rb") as f:
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

    data_transforms = TrainTransforms()
    valid_transforms = ValidationTransforms()
    train_dicts, valid_dicts = split_train_valid(
        data_dicts, valid_portion=valid_portion
    )
    train_set = Dataset(train_dicts, transform=data_transforms.transforms)
    valid_set = Dataset(valid_dicts, transform=valid_transforms.transforms)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=5
    )
    valid_loader = DataLoader(valid_set, batch_size=1)
    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
    }

    return dataloaders
