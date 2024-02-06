import argparse

default_common_args = {'num_cl': 100, 'max_mal': 10,
                       'attack': 'no_attack', 'scale_factor': 10,
                       'check': 'no_check', 'pred': 'l2norm',
                       'gpu': 0,
                       'bit': 32}
default_args = {'organamnist': {'lr': 5.0, 'momentum': 0.9,
                                'local_lr': 0.01, 'noise_std': 0.1, 'label_1': 5, 'label_2': 9,
                                'norm_bound': 0.2,
                                'local_optimizer': 'sgd',
                                'local_batch_size': 32, 'local_epochs': 1, 'epochs': 50},
                'organsmnist': {'lr': 2.0, 'momentum': 0.0,
                                'local_lr': 0.01, 'noise_std': 10, 'label_1': 5, 'label_2': 9,
                                'norm_bound_nn': 25.0, 'norm_bound_running_mean': 2000,
                                'norm_bound_running_var': 500000,
                                'local_optimizer': 'adam',
                                'local_batch_size': 256, 'local_epochs': 10, 'epochs': 100},
                'forest': {'lr': 2.0, 'momentum': 0.0,
                           'local_lr': 0.02, 'noise_std': 10, 'label_1': 1, 'label_2': 5,
                           'norm_bound_nn': 25.0, 'norm_bound_running_mean': 100,
                           'norm_bound_running_var': 500000,
                           'local_optimizer': 'adam',
                           'local_batch_size': 16384, 'local_epochs': 10, 'epochs': 200}}
for dataset in default_args:
    default_args[dataset].update(default_common_args)

def has_batch_norm_layer(dataset):
    return dataset in ['organsmnist', 'forest']

def parse_args(default_args=default_args):
    parser = argparse.ArgumentParser(description="Experiment with attack and defense")
    parser.add_argument("--dataset", type=str, help="dataset to run experiment on: 'organamnist', 'organsmnist', 'forest'", default='organamnist')
    args, remaining_argv = parser.parse_known_args()
    default_args_on_dataset = default_args[args.dataset]

    parser.add_argument("--lr", type=float, help="global learning rate", default=default_args_on_dataset['lr'])
    parser.add_argument("--momentum", type=float, help="global learning momentum", default=default_args_on_dataset['momentum'])
    parser.add_argument("--local-lr", type=float, help="local learning rate", default=default_args_on_dataset['local_lr'])

    parser.add_argument("--num-cl", type=int, help="number of clients", default=default_args_on_dataset['num_cl'])
    parser.add_argument("--max-mal", type=int, help="maximum number of malicious clients", default=default_args_on_dataset['max_mal'])

    parser.add_argument("--attack", type=str, help="attack type: 'no_attack', 'scale', 'noise', 'flip'", default=default_args_on_dataset['attack'])
    parser.add_argument("--scale-factor", type=float, help="scale factor if attack type is 'no_attack'", default=default_args_on_dataset['scale_factor'])
    parser.add_argument("--noise-std", type=float, help="noise std if attack type is 'noise'", default=default_args_on_dataset['noise_std'])
    parser.add_argument("--label-1", type=int, help="the label to change from if attack type is 'flip'", default=default_args_on_dataset['label_1'])
    parser.add_argument("--label-2", type=int, help="the label to change to if attack type is 'flip'", default=default_args_on_dataset['label_2'])

    parser.add_argument("--check", type=str, help="check type: 'no_check', 'strict', 'prob_zkp'", default=default_args_on_dataset['check'])
    parser.add_argument("--pred", type=str, help="check predicate: 'l2norm', 'sphere', 'cosine'", default=default_args_on_dataset['pred'])

    if not has_batch_norm_layer(args.dataset):
        parser.add_argument("--norm-bound", type=float, help="l2 norm bound of l2norm check or cosine check", default=default_args_on_dataset['norm_bound'])
    else:
        parser.add_argument("--norm-bound-nn", type=float, help="nn l2 norm bound of l2norm check or cosine check", default=25.0)
        parser.add_argument("--norm-bound-running-mean", type=float, help="running mean l2 norm bound of l2norm check or cosine check", default=100000000)
        parser.add_argument("--norm-bound-running-var", type=float, help="running var l2 norm bound of l2norm check or cosine check", default=100000000)

    parser.add_argument("--local-optimizer", type=str, help="local optimizer", default=default_args_on_dataset['local_optimizer'])
    parser.add_argument("--local-batch-size", type=int, help="local batch size", default=default_args_on_dataset['local_batch_size'])
    parser.add_argument("--local-epochs", type=int, help="number of local epochs", default=default_args_on_dataset['local_epochs'])
    parser.add_argument("--epochs", type=int, help="number of epochs", default=default_args_on_dataset['epochs'])

    parser.add_argument("--gpu", type=int, help="gpu number", default=default_args_on_dataset['gpu'])
    parser.add_argument("--bit", type=int, help="bit length", default=default_args_on_dataset['bit'])

    args2 = parser.parse_args(remaining_argv)
    args2.dataset = args.dataset

    return args2

def print_args(args):
    print("dataset:", args.dataset)
    if args.dataset == 'organamnist':
        print("model: CNN")
    elif args.dataset == 'organsmnist':
        print("model: ResNet18")
    elif args.dataset == 'forest':
        print("model: TabNet")
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
        if not has_batch_norm_layer(args.dataset):
            print("check l2 norm bound:", args.norm_bound)
        else:
            print("check nn l2 norm bound:", args.norm_bound_nn)
            print("check running mean l2 norm bound:", args.norm_bound_running_mean)
            print("check running var l2 norm bound:", args.norm_bound_running_var)

    print("local batch size:", args.local_batch_size)
    print("local epochs:", args.local_epochs)
    print("epochs:", args.epochs)

    print("run on gpu:", args.gpu)
    print("bit length:", args.bit)
