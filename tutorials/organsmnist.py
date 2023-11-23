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

# Create a parser
parser = argparse.ArgumentParser(description="OrganAMNIST with CNN")

# Define arguments
parser.add_argument("--lr", type=float, help="global learning rate", default=1.0)
parser.add_argument("--local-lr", type=float, help="local learning rate", default=0.01)

parser.add_argument("--num-cl", type=int, help="number of clients", default=100)
parser.add_argument("--max-mal", type=int, help="maximum number of malicious clients", default=10)

parser.add_argument("--attack", type=str, help="attack type: 'no_attack', 'scale', 'noise', 'flip'", default="no_attack")
parser.add_argument("--scale-factor", type=float, help="scale factor if attack type is 'no_attack'", default=20)
parser.add_argument("--noise-std", type=float, help="noise std if attack type is 'noise'", default=0.1)
parser.add_argument("--label-1", type=int, help="the label to change from if attack type is 'flip'", default=5)
parser.add_argument("--label-2", type=int, help="the label to change to if attack type is 'flip'", default=9)

parser.add_argument("--check", type=str, help="check type: 'no_check', 'strict', 'prob_zkp'", default="no_check")
parser.add_argument("--pred", type=str, help="check predicate: 'l2norm', 'sphere', 'cosine'", default="l2norm")
parser.add_argument("--norm-bound-nn", type=float, help="nn l2 norm bound of l2norm check or cosine check", default=1.0)
parser.add_argument("--norm-bound-running-mean", type=float, help="running mean l2 norm bound of l2norm check or cosine check", default=100000000)
parser.add_argument("--norm-bound-running-var", type=float, help="running var l2 norm bound of l2norm check or cosine check", default=100000000)


parser.add_argument("--local-batch-size", type=int, help="local batch size", default=32)
parser.add_argument("--local-epochs", type=int, help="number of local epochs", default=10)
parser.add_argument("--epochs", type=int, help="number of epochs", default=100)

parser.add_argument("--gpu", type=int, help="gpu number", default=0)

# Parse the command line arguments
args = parser.parse_args()

print("global lr:", args.lr)
print("local lr:", args.local_lr)
print("number of clients:", args.num_cl)
print("max number of malicious clients:", args.max_mal)
print("attack type:", args.attack)
if args.attack == 'no_attack':
    pass
elif args.attack == 'scale':
    print("scale factor:", args.scale_factor)
elif args.attack == 'noise':
    print("noise std:", args.noise_std)
elif args.attack == "flip":
    print("label 1:", args.label_1)
    print("label 2:", args.label_2)
else:
    raise ValueError("Incorrect attack type!")

print("check type:", args.check)
assert args.check in ['no_check', 'strict', 'prob_zkp']
if args.check != 'no_check':
    print("check pred:", args.pred)
    print("check nn l2 norm bound:", args.norm_bound_nn)
    print("check running mean l2 norm bound:", args.norm_bound_running_mean)
    print("check running var l2 norm bound:", args.norm_bound_running_var)

print("local batch size:", args.local_batch_size)
print("local epochs:", args.local_epochs)
print("epochs:", args.epochs)


USE_CUDA = True
LOCAL_BATCH_SIZE = args.local_batch_size
# EXAMPLES_PER_USER = 500
IMAGE_SIZE = 28

NUM_CLIENTS = args.num_cl
MAX_MALICIOUS_CLIENTS = args.max_mal

# suppress large outputs
VERBOSE = False

train_dataset_orig = OrganSMNIST(split="train", download=True)
test_dataset_orig = OrganSMNIST(split="test", download=True)

train_images = torch.from_numpy(train_dataset_orig.imgs).float().view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
train_labels = torch.from_numpy(train_dataset_orig.labels)
test_images = torch.from_numpy(test_dataset_orig.imgs).float().view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
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
model = resnet18()
model.fc = torch.nn.Linear(512, 11)
#
# def freeze_batch_norm_layer(module):
#     module.track_running_stats = False
#
# def freeze_all_batch_norm_layers(model):
#     freeze_batch_norm_layer(model.bn1)
#     freeze_batch_norm_layer(model.layer1._modules['0'].bn1)
#     freeze_batch_norm_layer(model.layer1._modules['0'].bn2)
#     freeze_batch_norm_layer(model.layer1._modules['1'].bn1)
#     freeze_batch_norm_layer(model.layer1._modules['1'].bn2)
#     freeze_batch_norm_layer(model.layer2._modules['0'].bn1)
#     freeze_batch_norm_layer(model.layer2._modules['0'].bn2)
#     freeze_batch_norm_layer(model.layer2._modules['0'].downsample._modules['1'])
#     freeze_batch_norm_layer(model.layer2._modules['1'].bn1)
#     freeze_batch_norm_layer(model.layer2._modules['1'].bn2)
#     freeze_batch_norm_layer(model.layer3._modules['0'].bn1)
#     freeze_batch_norm_layer(model.layer3._modules['0'].bn2)
#     freeze_batch_norm_layer(model.layer3._modules['0'].downsample._modules['1'])
#     freeze_batch_norm_layer(model.layer3._modules['1'].bn1)
#     freeze_batch_norm_layer(model.layer3._modules['1'].bn2)
#     freeze_batch_norm_layer(model.layer4._modules['0'].bn1)
#     freeze_batch_norm_layer(model.layer4._modules['0'].bn2)
#     freeze_batch_norm_layer(model.layer4._modules['0'].downsample._modules['1'])
#     freeze_batch_norm_layer(model.layer4._modules['1'].bn1)
#     freeze_batch_norm_layer(model.layer4._modules['1'].bn2)
#
# from torch import nn
# nn.Module.freeze_all_batch_norm_layers = freeze_all_batch_norm_layers

# 2. Choose where the model will be allocated.
cuda_enabled = torch.cuda.is_available() and USE_CUDA
device = torch.device(f"cuda:{args.gpu}" if cuda_enabled else "cpu")

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
                # "momentum": 0.9
            },
            # type of user selection sampling
            "active_user_selector": {"_base_": "base_uniformly_random_active_user_selector"},
        },
        "client": {
            # number of client's local epoch
            "epochs": args.local_epochs,
            "optimizer": {
                "_base_": "base_optimizer_adam",
                # client's local learning rate
                "lr": args.local_lr,
                # client's local momentum
                # "momentum": 0,
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
    attack_param={'scale_factor': {'nn': args.scale_factor,
                                   # 'running_mean': 1.0,
                                   # 'running_var': 1.0
                                   },
                  'noise_std': args.noise_std,
                  'label_1': args.label_1,
                  'label_2': args.label_2},
    check_type=args.check,  # 'no_check', 'strict', 'prob_zkp'
    check_param={'pred': args.pred, # 'l2norm', 'sphere', 'cosine'
                 'norm_bound': {'nn': args.norm_bound_nn,
                                'running_mean': args.norm_bound_running_mean,
                                'running_var': args.norm_bound_running_var}},
)

# We can now test our trained model.
trainer.test(
    data_provider=data_provider,
    metrics_reporter=MetricsReporter([Channel.STDOUT]),
)


