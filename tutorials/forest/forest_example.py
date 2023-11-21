#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pytorch_tabnet.tab_model import TabNetClassifier

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(0)


import os
import wget
from pathlib import Path
import shutil
import gzip

from matplotlib import pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# # Download ForestCoverType dataset

# In[2]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
dataset_name = 'forest-cover-type'
tmp_out = Path('./data/'+dataset_name+'.gz')
out = Path(os.getcwd()+'/data/'+dataset_name+'.csv')


# In[3]:


out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    print("File already exists.")
else:
    print("Downloading file...")
    wget.download(url, tmp_out.as_posix())
    with gzip.open(tmp_out, 'rb') as f_in:
        with open(out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    


# # Load data and split
# Same split as in original paper

# In[4]:


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


# In[5]:


train = pd.read_csv(out, header=None, names=feature_columns)

n_total = len(train)

# Train, val and test split follows
# Rory Mitchell, Andrey Adinets, Thejaswi Rao, and Eibe Frank.
# Xgboost: Scalable GPU accelerated learning. arXiv:1806.11248, 2018.

train_val_indices, test_indices = train_test_split(
    range(n_total), test_size=0.2, random_state=0)
train_indices, valid_indices = train_test_split(
    train_val_indices, test_size=0.2 / 0.6, random_state=0)


# # Simple preprocessing
# 
# Label encode categorical features and fill empty cells.

# In[6]:


categorical_columns = []
categorical_dims =  {}
for col in train.columns[train.dtypes == object]:
    print(col, train[col].nunique())
    l_enc = LabelEncoder()
    train[col] = train[col].fillna("VV_likely")
    train[col] = l_enc.fit_transform(train[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)

for col in train.columns[train.dtypes == 'float64']:
    train.fillna(train.loc[train_indices, col].mean(), inplace=True)


# # Define categorical features for categorical embeddings

# In[7]:


# This is a generic pipeline but actually no categorical features are available for this dataset

unused_feat = []

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]



# # Training

# In[9]:

# change labels from 1...7 to 0...6
train[target] -= 1

if os.getenv("CI", False):
# Take only a subsample to run CI
    X_train = train[features].values[train_indices][:1000,:]
    y_train = train[target].values[train_indices][:1000]
else:
    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]


# In[10]:

import numpy as np
import math
import torch
from flsim.utils.example_utils import SimpleConvNet
from torch.utils.data import Dataset
from flsim.data.data_sharder import SequentialSharder, PowerLawSharder
from flsim.utils.example_utils import DataLoader, DataProvider

class ForestDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        assert self.features.size(0) == self.labels.size(0)
    def __len__(self):
        return self.features.size(0)
    def __getitem__(self, user_id):
        return self.features[user_id], self.labels[user_id]

train_dataset = ForestDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).view(-1, 1))
valid_dataset = ForestDataset(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).view(-1, 1))
test_dataset = ForestDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).view(-1, 1))

NUM_CLIENTS = 1
# 2. Create a sharder, which maps samples in the training data to clients.
sharder = SequentialSharder(examples_per_shard=math.ceil(train_dataset.__len__() / NUM_CLIENTS))
# sharder = PowerLawSharder(num_shards=NUM_CLIENTS, alpha=0.5)

LOCAL_BATCH_SIZE = 16384

# 3. Shard and batchify training, eval, and test data.
fl_data_loader = DataLoader(
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    test_dataset=test_dataset,
    sharder=sharder,
    batch_size=LOCAL_BATCH_SIZE,
    drop_last=False,
)


max_epochs = 100 if not os.getenv("CI", False) else 2


# In[ ]:


from pytorch_tabnet.augmentations import ClassificationSMOTE
aug = ClassificationSMOTE(p=0.2)

data_provider = DataProvider(fl_data_loader)
print(f"\nClients in total: {data_provider.num_train_users()}")

clf = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=1,
    lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params = {"gamma": 0.95,
                     "step_size": 20},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
)


import random
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.data.data_sharder import FLDataSharder, SequentialSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.interfaces.metrics_reporter import Channel
from flsim.interfaces.model import IFLModel
from flsim.metrics_reporter.tensorboard_metrics_reporter import FLMetricsReporter
from flsim.utils.data.data_utils import batchify
from flsim.utils.simple_batch_metrics import FLBatchMetrics
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm


# nn.Module is clf.network
# model = clf.network


# 2. Choose where the model will be allocated.
cuda_enabled = torch.cuda.is_available()
device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")

# model, device

from flsim.utils.example_utils import FLModel

class TabNetFLModel(FLModel):
    def __init__(self, tabnet_model):
        self.tabnet_model = tabnet_model

    def construct_network(
            self,
            X_train,
            y_train,
            eval_set=None,
            eval_name=None,
            eval_metric=None,
            loss_fn=None,
            weights=0,
            max_epochs=100,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=True,
            callbacks=None,
            pin_memory=True,
            from_unsupervised=None,
            warm_start=False,
            augmentations=None,
            compute_importance=True,
            device=torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    ):
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.

        Parameters
        ----------
        X_train : np.ndarray
            Train set
        y_train : np.array
            Train targets
        eval_set : list of tuple
            List of eval tuple set (X, y).
            The last one is used for early stopping
        eval_name : list of str
            List of eval set names.
        eval_metric : list of str
            List of evaluation metrics.
            The last metric is used for early stopping.
        loss_fn : callable or None
            a PyTorch loss function
        weights : bool or dictionnary
            0 for no balancing
            1 for automated balancing
            dict for custom weights per class
        max_epochs : int
            Maximum number of epochs during training
        patience : int
            Number of consecutive non improving epoch before early stopping
        batch_size : int
            Training batch size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
        num_workers : int
            Number of workers used in torch.utils.data.DataLoader
        drop_last : bool
            Whether to drop last batch during training
        callbacks : list of callback function
            List of custom callbacks
        pin_memory: bool
            Whether to set pin_memory to True or False during training
        from_unsupervised: unsupervised trained model
            Use a previously self supervised model as starting weights
        warm_start: bool
            If True, current model parameters are used to start training
        compute_importance : bool
            Whether to compute feature importance
        """

        from dataclasses import dataclass, field
        from typing import List, Any, Dict
        import torch
        from torch.nn.utils import clip_grad_norm_
        import numpy as np
        from scipy.sparse import csc_matrix
        from abc import abstractmethod
        from pytorch_tabnet import tab_network
        from pytorch_tabnet.utils import (
            SparsePredictDataset,
            PredictDataset,
            create_explain_matrix,
            validate_eval_set,
            create_dataloaders,
            define_device,
            ComplexEncoder,
            check_input,
            check_warm_start,
            create_group_matrix,
            check_embedding_parameters
        )
        from pytorch_tabnet.callbacks import (
            CallbackContainer,
            History,
            EarlyStopping,
            LRSchedulerCallback,
        )
        from pytorch_tabnet.metrics import MetricContainer, check_metrics
        from sklearn.base import BaseEstimator

        from torch.utils.data import DataLoader
        import io
        import json
        from pathlib import Path
        import shutil
        import zipfile
        import warnings
        import copy
        import scipy
        
        self.tabnet_model.max_epochs = max_epochs
        self.tabnet_model.patience = patience
        self.tabnet_model.batch_size = batch_size
        self.tabnet_model.virtual_batch_size = virtual_batch_size
        self.tabnet_model.num_workers = num_workers
        self.tabnet_model.drop_last = drop_last
        self.tabnet_model.input_dim = X_train.shape[1]
        self.tabnet_model._stop_training = False
        self.tabnet_model.pin_memory = pin_memory and (self.tabnet_model.device.type != "cpu")
        self.tabnet_model.augmentations = augmentations
        self.tabnet_model.compute_importance = compute_importance

        if self.tabnet_model.augmentations is not None:
            # This ensure reproducibility
            self.tabnet_model.augmentations._set_seed()

        eval_set = eval_set if eval_set else []

        if loss_fn is None:
            self.tabnet_model.loss_fn = self.tabnet_model._default_loss
        else:
            self.tabnet_model.loss_fn = loss_fn

        check_input(X_train)
        check_warm_start(warm_start, from_unsupervised)

        self.tabnet_model.update_fit_params(
            X_train,
            y_train,
            eval_set,
            weights,
        )

        # Validate and reformat eval set depending on training data
        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

        train_dataloader, valid_dataloaders = self.tabnet_model._construct_loaders(
            X_train, y_train, eval_set
        )

        if from_unsupervised is not None:
            # Update parameters to match self.tabnet_model pretraining
            self.tabnet_model.__update__(**from_unsupervised.get_params())

        if not hasattr(self.tabnet_model, "network") or not warm_start:
            # model has never been fitted before of warm_start is False
            self.tabnet_model._set_network()

        self.tabnet_model._update_network_params()
        self.tabnet_model._set_metrics(eval_metric, eval_names)
        self.tabnet_model._set_optimizer()
        self.tabnet_model._set_callbacks(callbacks)
        if from_unsupervised is not None:
            self.tabnet_model.load_weights_from_unsupervised(from_unsupervised)
            warnings.warn("Loading weights from unsupervised pretraining")

        # set FLModel's model to tabnet_model's network
        self.model = self.tabnet_model.network
        self.device = device


    # def post_construct_network(
    #         self,
    #         X_train,
    #         y_train,
    #         eval_set=None,
    #         eval_name=None,
    #         eval_metric=None,
    #         loss_fn=None,
    #         weights=0,
    #         max_epochs=100,
    #         patience=10,
    #         batch_size=1024,
    #         virtual_batch_size=128,
    #         num_workers=0,
    #         drop_last=True,
    #         callbacks=None,
    #         pin_memory=True,
    #         from_unsupervised=None,
    #         warm_start=False,
    #         augmentations=None,
    #         compute_importance=True
    # ):
    #
    #     # Call method on_train_begin for all callbacks
    #     self._callback_container.on_train_begin()
    #
    #     # Training loop over epochs
    #     for epoch_idx in range(self.max_epochs):
    #
    #         # Call method on_epoch_begin for all callbacks
    #         self._callback_container.on_epoch_begin(epoch_idx)
    #
    #         self._train_epoch(train_dataloader)
    #
    #         # Apply predict epoch to all eval sets
    #         for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
    #             self._predict_epoch(eval_name, valid_dataloader)
    #
    #         # Call method on_epoch_end for all callbacks
    #         self._callback_container.on_epoch_end(
    #             epoch_idx, logs=self.history.epoch_metrics
    #         )
    #
    #         if self._stop_training:
    #             break
    #
    #     # Call method on_train_end for all callbacks
    #     self._callback_container.on_train_end()
    #     self.network.eval()
    #
    #     if self.compute_importance:
    #         # compute feature importance once the best model is defined
    #         self.feature_importances_ = self._compute_feature_importances(X_train)


    def fl_forward(self, batch) -> FLBatchMetrics:

        features = batch["features"]  # [B, C, 28, 28]
        batch_label = batch["labels"]

        stacked_label = batch_label.view(-1).long().clone().detach()
        if self.device is not None:
            features = features.to(self.device)

        X = features
        y = batch_label
        
        X = X.to(self.tabnet_model.device).float()
        y = y.to(self.tabnet_model.device).float()

        if self.tabnet_model.augmentations is not None:
            X, y = self.tabnet_model.augmentations(X, y)

        for param in self.tabnet_model.network.parameters():
            param.grad = None

        output, M_loss = self.tabnet_model.network(X)

        loss = self.tabnet_model.compute_loss(output, y)
        # Add the overall sparsity loss
        loss = loss - self.tabnet_model.lambda_sparse * M_loss

        # Perform backward pass and optimization
        # loss.backward()
        # if self.tabnet_model.clip_value:
        #     from torch.nn.utils import clip_grad_norm_
        #     clip_grad_norm_(self.tabnet_model.network.parameters(), self.tabnet_model.clip_value)
        # self.tabnet_model._optimizer.step()

        # batch_logs["loss"] = loss.cpu().detach().numpy().item()
        #
        # return batch_logs


        # output = self.model(features)

        if self.device is not None:
            output, batch_label, stacked_label = (
                output.to(self.device),
                batch_label.to(self.device),
                stacked_label.to(self.device),
            )

        # loss = F.cross_entropy(output, stacked_label)
        num_examples = self.get_num_examples(batch)
        output = output.detach().cpu()
        stacked_label = stacked_label.detach().cpu()
        del features
        return FLBatchMetrics(
            loss=loss,
            num_examples=num_examples,
            predictions=output,
            targets=stacked_label,
            model_inputs=[],
        )


# 3. Wrap the model with TabNetFLModel.
global_model = TabNetFLModel(clf)
global_model.construct_network(X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    max_epochs=max_epochs, patience=100,
    batch_size=16384, virtual_batch_size=256,
    augmentations=aug,
    device=device)

# assert(global_model.fl_get_module() == model)

# 4. Move the model to GPU and enable CUDA if desired.
if cuda_enabled:
    global_model.fl_cuda()

from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.example_utils import MetricsReporter

# Create a metric reporter.
metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

import inspect
#
# if VERBOSE:
#     print(inspect.getsource(MetricsReporter.compute_scores))
#     print(inspect.getsource(MetricsReporter.create_eval_metrics))
#     print(inspect.getsource(MetricsReporter.compare_metrics))

import flsim.configs
from flsim.utils.config_utils import fl_config_from_json
from omegaconf import OmegaConf

json_config = {
    "trainer": {
        "_base_": "base_sync_trainer",
        # there are different types of aggregator
        # fed avg doesn't require lr, while others such as fed_avg_with_lr or fed_adam do
        "_base_": "base_sync_trainer",
        "server": {
            "_base_": "base_sync_server",
            "server_optimizer": {
                "_base_": "base_fed_avg_with_lr",
                # "lr": args.lr,
                "lr": 1.0,
                "momentum": 0.
            },
            # type of user selection sampling
            "active_user_selector": {"_base_": "base_uniformly_random_active_user_selector"},
        },
        "client": {
            # number of client's local epoch
            "epochs": 1,
            "optimizer": {
                "_base_": "base_optimizer_sgd",
                # client's local learning rate
                # "lr": args.local_lr,
                "lr": 0.02,
                # client's local momentum
                "momentum": 0,
            },
        },
        # number of users per round for aggregation
        "users_per_round": 1,
        # "users_per_round": NUM_CLIENTS,
        # total number of global epochs
        # total #rounds = ceil(total_users / users_per_round) * epochs
        # "epochs": args.epochs,
        "epochs": 100,
        # frequency of reporting train metrics
        "train_metrics_reported_per_epoch": 100,
        # frequency of evaluation per epoch
        "eval_epoch_frequency": 1,
        "do_eval": True,
        # should we report train metrics after global aggregation
        "report_train_metrics_after_aggregation": True,
    }
}
cfg = fl_config_from_json(json_config)

from hydra.utils import instantiate

# Instantiate the trainer.
trainer = instantiate(cfg.trainer, model=global_model, cuda_enabled=cuda_enabled)

# Launch FL training.
final_model, eval_score = trainer.train(
    data_provider=data_provider,
    metrics_reporter=metrics_reporter,
    num_total_users=data_provider.num_train_users(),
    distributed_world_size=1,
    # malicious_count=MAX_MALICIOUS_CLIENTS,
    malicious_count=0,
    # attack_type=args.attack,  # 'scale', 'noise', 'flip'
    attack_type='noise',
    attack_param={},
    # attack_param={'scale_factor': args.scale_factor,
    #               'noise_std': args.noise_std,
    #               'label_1': args.label_1,
    #               'label_2': args.label_2},
    # check_type=args.check,  # 'no_check', 'strict', 'prob_zkp'
    check_type='no_check',
    check_param={'pred': 'l2norm',
                 'norm_bound': 1e8},
    # check_param={'pred': args.pred, # 'l2norm', 'sphere', 'cosine'
    #              'norm_bound': args.norm_bound},
)


# clf.fit(
#     X_train=X_train, y_train=y_train,
#     eval_set=[(X_train, y_train), (X_valid, y_valid)],
#     eval_name=['train', 'valid'],
#     max_epochs=max_epochs, patience=100,
#     batch_size=16384, virtual_batch_size=256,
#     augmentations=aug
# )
#
