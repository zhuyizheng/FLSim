import numpy as np
import torch
from flsim.utils.example_utils import SimpleConvNet
from medmnist import OrganAMNIST, OrganSMNIST
from torch.utils.data import Dataset
from flsim.data.data_sharder import SequentialSharder, PowerLawSharder
from flsim.utils.example_utils import DataLoader, DataProvider

USE_CUDA = True
LOCAL_BATCH_SIZE = 32
EXAMPLES_PER_USER = 500
IMAGE_SIZE = 28

# suppress large outputs
VERBOSE = False

train_dataset_orig = OrganAMNIST(split="train", download=True)
test_dataset_orig = OrganAMNIST(split="test", download=True)
# dataset_smnist = OrganSMNIST(split="test", download=True)

train_images = torch.from_numpy(train_dataset_orig.imgs).float().view(-1, 1, 28, 28)
train_labels = torch.from_numpy(train_dataset_orig.labels)
test_images = torch.from_numpy(test_dataset_orig.imgs).float().view(-1, 1, 28, 28)
test_labels = torch.from_numpy(test_dataset_orig.labels)

class organDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        assert self.images.size(0) == self.labels.size(0)
    def __len__(self):
        return self.images.size(0)
    def __getitem__(self, user_id):
        return self.images[user_id], self.labels[user_id]

train_dataset = organDataset(train_images, train_labels)
test_dataset = organDataset(test_images, test_labels)

NUM_CLIENTS = 10
MAX_MALICIOUS_CLIENTS = 1
# 2. Create a sharder, which maps samples in the training data to clients.
# sharder = SequentialSharder(examples_per_shard=EXAMPLES_PER_USER)
sharder = PowerLawSharder(num_shards=NUM_CLIENTS, alpha=0.5)

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
model = SimpleConvNet(in_channels=1, num_classes=11)

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
                "lr": 2.13,
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
                "lr": 0.01,
                # client's local momentum
                "momentum": 0,
            },
        },
        # number of users per round for aggregation
        "users_per_round": NUM_CLIENTS,
        # total number of global epochs
        # total #rounds = ceil(total_users / users_per_round) * epochs
        "epochs": 10,
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
    malicious_count=1,
    attack_type='noise',  # 'scale', 'noise', 'flip'
    attack_param={'scale_factor': -1.5,
                  'noise_std': 0.1,
                  'label_1': 5,
                  'label_2': 9},
    check_type='strict',  # 'no_check', 'strict', 'prob_zkp'
    check_param={'pred': 'l2norm', # 'l2norm', 'sphere', 'cosine'
                 'norm_bound': 0.2},
)

# We can now test our trained model.
trainer.test(
    data_provider=data_provider,
    metrics_reporter=MetricsReporter([Channel.STDOUT]),
)


