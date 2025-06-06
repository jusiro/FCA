"""
Dataset and Dataloaser preparation for vision-language for generalization/transferability.
It contains utils for balancing datasets with regard the categories.
"""

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from data.dataset import Dataset
from torchvision.transforms import Compose

from .transforms import LoadImage


def set_loader(dataframe_path, data_root_path, targets, batch_size=64, num_workers=6, size=224, norm=True):

    # Prepare data transforms for pre-processing
    transforms = Compose([LoadImage(size=(size, size), canvas=True, norm=norm)])

    # Load data from dict file in txt
    data = []
    dataframe = pd.read_csv(dataframe_path)
    for i in range(len(dataframe)):
        sample_df = dataframe.loc[i, :].to_dict()
        labels = [sample_df[iTarget] for iTarget in targets]
        if np.sum(labels) == 1:
            # Image and label
            data_i = {"image_path": data_root_path + sample_df["Path"],
                      "label": int(np.argmax([sample_df[iTarget] for iTarget in targets]))}
            data.append(data_i)

    # Set dataset and data loader
    loader = get_loader(data, transforms, batch_size, num_workers)

    return loader


def get_loader(data, transforms, batch_size, num_workers):
    if len(data) == 0:
        loader = None
    else:
        dataset = Dataset(data=data, transform=transforms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return loader