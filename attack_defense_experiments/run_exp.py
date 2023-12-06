# reused code from https://github.com/zhuyizheng/FLSim/blob/main/tutorials/cifar10_tutorial.ipynb
# original in https://github.com/facebookresearch/FLSim/blob/main/tutorials/cifar10_tutorial.ipynb

import numpy as np
import math
import torch
from flsim.utils.example_utils import SimpleConvNet
from medmnist import OrganAMNIST, OrganSMNIST
from torch.utils.data import Dataset
from flsim.data.data_sharder import SequentialSharder, PowerLawSharder
from flsim.utils.example_utils import DataLoader, DataProvider
from flsim.utils.example_utils import FLModel
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.example_utils import MetricsReporter
import inspect
import flsim.configs
from flsim.utils.config_utils import fl_config_from_json
from omegaconf import OmegaConf
from hydra.utils import instantiate

from experiment_utils.parser import parse_args, print_args, has_batch_norm_layer
from experiment_utils.args_to_json import args_to_json
from experiment_utils.gen_dataset import gen_dataset
from experiment_utils.gen_global_model import gen_global_model

args = parse_args()
print_args(args)

USE_CUDA = True

train_dataset, test_dataset, extra_info = gen_dataset(args.dataset)

sharder = SequentialSharder(examples_per_shard=math.ceil(train_dataset.__len__() / args.num_cl))

fl_data_loader = DataLoader(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    test_dataset=test_dataset,
    sharder=sharder,
    batch_size=args.local_batch_size,
    drop_last=False,
)

data_provider = DataProvider(fl_data_loader)
print(f"\nClients in total: {data_provider.num_train_users()}")

cuda_enabled = torch.cuda.is_available() and USE_CUDA

global_model = gen_global_model(args, extra_info, USE_CUDA)

metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

json_config = args_to_json(args)
cfg = fl_config_from_json(json_config)

trainer = instantiate(cfg.trainer, model=global_model, cuda_enabled=cuda_enabled)

final_model, eval_score = trainer.train(
    data_provider=data_provider,
    metrics_reporter=metrics_reporter,
    num_total_users=data_provider.num_train_users(),
    distributed_world_size=1,
    malicious_count=args.max_mal,
    attack_type=args.attack,  # 'scale', 'noise', 'flip'
    attack_param={'scale_factor': args.scale_factor,
                  'noise_std': args.noise_std,
                  'label_1': args.label_1,
                  'label_2': args.label_2},
    check_type=args.check,  # 'no_check', 'strict', 'prob_zkp'
    check_param={'pred': args.pred, # 'l2norm', 'sphere', 'cosine'
                 'norm_bound':
                     args.norm_bound if not has_batch_norm_layer(args.dataset) else {
                     'nn': args.norm_bound_nn,
                     'running_mean': args.norm_bound_running_mean,
                     'running_var': args.norm_bound_running_var},
                 },
)

trainer.test(
    data_provider=data_provider,
    metrics_reporter=MetricsReporter([Channel.STDOUT]),
)

