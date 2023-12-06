# uses code from https://github.com/dreamquark-ai/tabnet/blob/develop/forest_example.ipynb


import torch
from medmnist import OrganAMNIST, OrganSMNIST
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import os
import wget
from pathlib import Path
import shutil
import gzip

class TorchDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        assert self.images.size(0) == self.labels.size(0)

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, user_id):
        return self.images[user_id], self.labels[user_id]


def organ_dataset_from_orig(dataset_orig, input_channels=1):
    IMAGE_SIZE = 28
    images = torch.from_numpy(dataset_orig.imgs).float().view(-1, 1, IMAGE_SIZE, IMAGE_SIZE).repeat(1, input_channels, 1, 1)
    labels = torch.from_numpy(dataset_orig.labels)
    return TorchDataset(images, labels)

def gen_dataset_organamnist():
    train_dataset_orig = OrganAMNIST(split="train", download=True)
    test_dataset_orig = OrganAMNIST(split="test", download=True)
    return organ_dataset_from_orig(train_dataset_orig), organ_dataset_from_orig(test_dataset_orig), None


def gen_dataset_organsmnist():
    train_dataset_orig = OrganSMNIST(split="train", download=True)
    test_dataset_orig = OrganSMNIST(split="test", download=True)
    return organ_dataset_from_orig(train_dataset_orig, input_channels=3), organ_dataset_from_orig(test_dataset_orig, input_channels=3), None

def gen_dataset_forest():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    dataset_name = 'forest-cover-type'
    tmp_out = Path('./data/' + dataset_name + '.gz')
    out = Path(os.getcwd() + '/data/' + dataset_name + '.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        print("Downloading file...")
        wget.download(url, tmp_out.as_posix())
        with gzip.open(tmp_out, 'rb') as f_in:
            with open(out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    target = "Covertype"

    bool_columns = [
        "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
        "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
        "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
        "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
        "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
        "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
        "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
        "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
        "Soil_Type40"
    ]

    int_columns = [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
    ]

    feature_columns = (
        int_columns + bool_columns + [target])

    train = pd.read_csv(out, header=None, names=feature_columns)

    n_total = len(train)

    train_indices, test_indices = train_test_split(
        range(n_total), test_size=0.2, random_state=0)


    categorical_columns = []
    categorical_dims = {}
    for col in train.columns[train.dtypes == object]:
        print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)

    for col in train.columns[train.dtypes == 'float64']:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)

    unused_feat = []

    features = [ col for col in train.columns if col not in unused_feat+[target]]

    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    train[target] -= 1

    if os.getenv("CI", False):
        # Take only a subsample to run CI
        X_train = train[features].values[train_indices][:1000, :]
        y_train = train[target].values[train_indices][:1000]
    else:
        X_train = train[features].values[train_indices]
        y_train = train[target].values[train_indices]

    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]

    train_dataset = TorchDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).view(-1, 1))
    test_dataset = TorchDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).view(-1, 1))

    extra_info = {'X_train': X_train,
                  'y_train': y_train,
                  'X_test': X_test,
                  'y_test': y_test,
                  'cat_idxs': cat_idxs,
                  'cat_dims': cat_dims}

    return train_dataset, test_dataset, extra_info

dataset_call_dict = {'organamnist': gen_dataset_organamnist,
                     'organsmnist': gen_dataset_organsmnist,
                     'forest': gen_dataset_forest}

def gen_dataset(dataset):
    return dataset_call_dict[dataset]()
