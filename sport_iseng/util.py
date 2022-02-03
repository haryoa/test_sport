"""
Every utils here!
"""

import os
from glob import glob
from typing import Dict, List

import pandas as pd


def gather_image_path(glob_path: str) -> List[str]:
    """
    Gather image path and return the path there

    Parameters
    ----------
    glob_path : str
        Glob path

    Returns
    -------
    List[str]
        Return path of desired image
    """
    return sorted(glob(glob_path))


def get_image_paths_with_labels(
    csv_file: str, base_dir: str, extracted_dataset: List[str]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Get image paths with labels

    Parameters
    ----------
    csv_file : str
        csv contains filepaths,labels,data set column
    base_dir : str
        Base directory of the label
    extract_dataset : List[str]
        List of `data set` column that wwant to be extracted.
        It will be shown in the return
        e.g.: train, test
        KEY:
        train: image_paths, labels
        test: image_paths, labels

    Returns
    -------
    Dict[str, Dict[str, str]]
        Return dictionary stated like in extract_dataset
        arguments
    """
    returned_dict = {}
    df = pd.read_csv(csv_file)  # pylint: disable=invalid-name

    for dataset in extracted_dataset:
        filtered = df[df["data set"] == dataset]
        imgs_list = filtered["filepaths"].apply(lambda x: os.path.join(base_dir, x))
        labels = filtered["labels"]
        returned_dict[dataset] = {
            "image_paths": imgs_list.tolist(),
            "labels": labels.tolist(),
        }
    return returned_dict
