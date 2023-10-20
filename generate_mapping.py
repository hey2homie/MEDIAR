import re
import argparse
import glob
import json
import os

import numpy as np


def map_files(path: str, image_filter: str, mask_filter: str, pattern: str) -> dict:
    """
    Map paths of images to corresponding labels to a dictionary. Assumes that the naming is of the type /123_image.png
    Args:
        path (str): Path to the folder containing the images and labels sub-folders.
        image_filter (str): Filter for the images.
        mask_filter (str): Filter for the labels.
        pattern (str, optional): Regex pattern for extracting the file number
    Return:
        dict: Dictionary containing the mapping.
    Raises:
        ValueError: If no images and/or labels are found in the path.
        ValueError: Cannot map image(s) to label(s).
    """
    if not glob.glob(os.path.join(path, "*/*")):
        raise ValueError("No images and/or labels found in path: {}".format(path))

    pattern = re.compile(pattern)
    get_img_number = np.vectorize(lambda x: pattern.search(x).group(1))
    get_data_item = np.vectorize(lambda x, y: {"img": x, "label": y})

    images = np.sort(
        np.array(glob.glob(os.path.join(path, "images/" + f"*{image_filter}*")))
    )
    labels = np.sort(
        np.array(glob.glob(os.path.join(path, "labels/" + f"*{mask_filter}*")))
    )

    try:
        if not np.all(get_img_number(images) == get_img_number(labels)):
            raise AssertionError("Cannot map image(s) to label(s)")
    except ValueError:
        raise ValueError("Number of images and labels do not match")

    data_dicts = get_data_item(images, labels).tolist()
    map_dict = {"data": data_dicts}
    return map_dict


def add_mapping_to_json(json_file: str, map_dict: dict) -> None:
    """
    Save mapped dictionary as a json file
    Args:
        json_file (str): Path to the json file.
        map_dict (dict): Dictionary containing the mapping.
    Return:
        None
    """
    with open(json_file, "w") as file:
        json.dump(map_dict, file)


if __name__ == "__main__":
    """
    Example of usage:
        python generate_mapping.py -p ~/Work/Dataset/ -i _rgb -l _masks
    """
    parser = argparse.ArgumentParser(description="Mapping files and paths")
    parser.add_argument("-p", "--path", default=".", type=str, help="Path to data")
    parser.add_argument(
        "-n",
        "--name",
        default="mapping.json",
        type=str,
        help="Name of the mapping file",
    )
    parser.add_argument(
        "-i", "--image_filter", default="", type=str, help="Filter for the images"
    )
    parser.add_argument(
        "-l", "--label_filter", default="", type=str, help="Filter for the labels"
    )
    parser.add_argument(
        "-r",
        "--regex",
        default=r"(\d+)_",
        type=str,
        help="Regex pattern for extracting the file number",
    )
    args = parser.parse_args()

    MAP_DIR = "./config/"
    map_labeled = os.path.join(MAP_DIR, "mapping_labeled.json")
    mapping = map_files(args.path, args.image_filter, args.label_filter, args.regex)
    add_mapping_to_json(map_labeled, mapping)
