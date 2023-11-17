import numpy as np
import math
import torch
from flsim.utils.example_utils import SimpleConvNet
from medmnist import OrganAMNIST, OrganSMNIST
from torch.utils.data import Dataset
from flsim.data.data_sharder import SequentialSharder, PowerLawSharder
from flsim.utils.example_utils import DataLoader, DataProvider
from torchvision.models import resnet18

import argparse

from torch import nn
import torch.nn.functional as F


from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

data_file = 'airbnb/AB_NYC_2019.csv'
df = pd.read_csv(data_file)

X = df.drop(['price', 'name', 'host_name'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def process_dataframe(df):
    # Fill NaN with something better?
    df.fillna(0, inplace=True)
    # columns_to_discr = [('age', 10), ('fnlwgt', 25), ('capital-gain', 10), ('capital-loss', 10),
    #                     ('hours-per-week', 10)]

    # def discretize_colum(data_clm, num_values=10):
    #     """ Discretize a column by quantiles """
    #     r = np.argsort(data_clm)
    #     bin_sz = (len(r) / num_values) + 1  # make sure all quantiles are in range 0-(num_quarts-1)
    #     q = r // bin_sz
    #     return q
    #
    # for clm, nvals in columns_to_discr:
    #     df[clm] = discretize_colum(df[clm], num_values=nvals)
    #     df[clm] = df[clm].astype(int).astype(str)

    # df['education_num'] = df['education_num'].astype(int).astype(str)
    # args.cat_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    X = df[features].to_numpy()
    y = df[label].to_numpy()

    return X, y

X_train, y_train = process_dataframe(df_train)
X_test, y_test = process_dataframe(df_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

num_features = 14
cat_idx = [1,3,5,6,7,8,9,13]
# cat_dims_from_cat_idx = [9, 16, 7, 15, 6, 5, 2, 42] # not needed. set automatically
cat_dims = []
num_idx = []

for i in range(num_features):
    if i in cat_idx:
        le = LabelEncoder()
        X_train[:, i] = le.fit_transform(X_train[:, i])
        X_test[:, i] = le.transform(X_test[:, i])

        # Setting this?
        cat_dims.append(len(le.classes_))

    else:
        num_idx.append(i)

scaler = StandardScaler()
X_train[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
X_test[:, num_idx] = scaler.transform(X_test[:, num_idx])


ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
new_x1 = ohe.fit_transform(X_train[:, cat_idx])
new_x2 = X_train[:, num_idx]
X_train = np.concatenate([new_x1, new_x2], axis=1)
# print("New Shape:", X_train.shape)

new_x1_test = ohe.transform(X_test[:, cat_idx])
new_x2_test = X_test[:, num_idx]
X_test = np.concatenate([new_x1_test, new_x2_test], axis=1)
# print("New Test Shape:", X_test.shape)


# X_train = X_train.astype('float')
# X_test = X_test.astype('float')
# print(X_train)

train_features = torch.from_numpy(X_train.astype('float32'))
train_labels = torch.from_numpy(y_train.astype('float32'))
test_features = torch.from_numpy(X_test.astype('float32'))
test_labels = torch.from_numpy(y_test.astype('float32'))

class AdultDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        assert self.features.size(0) == self.labels.size(0)
    def __len__(self):
        return self.features.size(0)
    def __getitem__(self, user_id):
        return self.features[user_id], self.labels[user_id]

train_dataset = AdultDataset(train_features, train_labels)
test_dataset = AdultDataset(test_features, test_labels)

# 2. Create a sharder, which maps samples in the training data to clients.
sharder = SequentialSharder(examples_per_shard=math.ceil(train_dataset.__len__() / NUM_CLIENTS))
# sharder = PowerLawSharder(num_shards=NUM_CLIENTS, alpha=0.5)

# 3. Shard and batchify training, eval, and test data.
fl_data_loader = DataLoader(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    test_dataset=test_dataset,
    sharder=sharder,
    batch_size=LOCAL_BATCH_SIZE,
    drop_last=False,
)

# 4. Wrap the data loader with a data provider.
data_provider = DataProvider(fl_data_loader)
print(f"\nClients in total: {data_provider.num_train_users()}")


# 1. Define our model, a simple CNN.
# model = SimpleConvNet(in_channels=1, num_classes=11)
model = MLP_Model(n_layers=4, input_dim=X_train.shape[1], hidden_dim=47, output_dim=2)


# 2. Choose where the model will be allocated.
cuda_enabled = torch.cuda.is_available() and USE_CUDA
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
                "lr": args.lr,
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
                "lr": args.local_lr,
                # client's local momentum
                "momentum": 0,
            },
        },
        # number of users per round for aggregation
        "users_per_round": NUM_CLIENTS,
        # total number of global epochs
        # total #rounds = ceil(total_users / users_per_round) * epochs
        "epochs": args.epochs,
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
if VERBOSE: print(OmegaConf.to_yaml(cfg))

from hydra.utils import instantiate

# Instantiate the trainer.
trainer = instantiate(cfg.trainer, model=global_model, cuda_enabled=cuda_enabled)

# Launch FL training.
final_model, eval_score = trainer.train(
    data_provider=data_provider,
    metrics_reporter=metrics_reporter,
    num_total_users=data_provider.num_train_users(),
    distributed_world_size=1,
    malicious_count=MAX_MALICIOUS_CLIENTS,
    attack_type=args.attack,  # 'scale', 'noise', 'flip'
    attack_param={'scale_factor': args.scale_factor,
                  'noise_std': args.noise_std,
                  'label_1': args.label_1,
                  'label_2': args.label_2},
    check_type=args.check,  # 'no_check', 'strict', 'prob_zkp'
    check_param={'pred': args.pred, # 'l2norm', 'sphere', 'cosine'
                 'norm_bound': args.norm_bound},
)

# We can now test our trained model.
trainer.test(
    data_provider=data_provider,
    metrics_reporter=MetricsReporter([Channel.STDOUT]),
)


