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


clf._set_network()
model = clf.network

# nn.Module is clf.network
# model = clf.network


# 2. Choose where the model will be allocated.
cuda_enabled = torch.cuda.is_available()
device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")

model, device

from flsim.utils.example_utils import FLModel

# 3. Wrap the model with FLModel.
global_model = FLModel(model, device)
assert(global_model.fl_get_module() == model)

# 4. Move the model to GPU and enable CUDA if desired.
if cuda_enabled:
    global_model.fl_cuda()

from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.example_utils import MetricsReporter

# Create a metric reporter.
metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

import inspect

if VERBOSE:
    print(inspect.getsource(MetricsReporter.compute_scores))
    print(inspect.getsource(MetricsReporter.create_eval_metrics))
    print(inspect.getsource(MetricsReporter.compare_metrics))

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
                "momentum": 0.9
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
    check_param={},
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
