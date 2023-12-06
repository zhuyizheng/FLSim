# A Fork of FLSim that simulates attacks and defenses in Federated Learning

This is a fork of [FLSim](https://github.com/facebookresearch/FLSim). 
The orgininal FLSim simulats Federated Learning on a single machine. This fork implements various attacks and defenses. Experiments can be run with 

```python attack_defense_experiments/run_exp.py [PARAMETERS]```

Description of parameters:
- `--dataset`: The dataset to run expirement on. Choose from `organamnist`, `organsmnist`, `forest`. Default: `organamnist`
- `--lr`: The global learning rate
- `--momentum`: The global momentum
- `--local-lr`: The local learning rate on each client
- `--num-cl`: Total number of clients to be simulated
- `--max-mal`: The number of malicious clients to be simulated
- `--attack`: The attack type. Choose from:
  - `no-attack`: no attack
  - `scale`: scale the model update by a constant, positive or negative
  - `noise`:add Gaussian noise to every coordinate
  - `flip`: change the label of one category to another
- `--scale-factor`: If `--attack scale`, the factor to scale the model update, positive or negative
- `--noise-std`: If `--attack noise`, the standard deviation of Gaussian noise to be added to each coordinate
- `--label-1`: If `--attack flip`, the label to change from
- `--label-2`: If `--attack flip`, the label to change to
- `--check`: The type of check on model updates. Choose from:
  - `no_check`: aggregate model updates from all the clients without checking
  - `strict`: perform strict checking
  - `prob_zkp`: perform probabilistic checking as in the paper https://arxiv.org/abs/2311.15310
- `--pred`: The check predicate. Choose from:
  - `l2norm`: check if L2 norm of model update is bounded by a specified value
  - `sphere`: check if the distance of a model update to a pivot vector is bounded by a specified value
  - `cosine`: check if the L2 norm of model update is bounded by a specified value and the cosine similarity of the model update and a pivot vector is bigger than a specified value
- `--norm-bound`: This parameter applies if `--dataset organamnist`. The corresponding CNN model does not have batch norm layers. If `--pred l2norm` or `--pred cosine`, the L2 norm bound of the model update; if `--pred sphere`, the distance bound
- `--norm-bound-nn`, `--norm-bound-running-mean`, `--norm-bound-running-var`: These parameters applies if `--dataset organsmnist` or `--dataset forest`. The corresponding ResNet18 model or TabNet model has batch norm layers. The model update of neural network parameters and `*_running_mean`, `*_running_var` tensors from batch norm layers must be treated separately. If `--pred l2norm` or `--pred cosine`, the L2 norm bound of the model update; if `--pred sphere`, the distance bound
- `--local-optimizer`: The optimizers used on each client's local training
- `--local-batch-size`: The batch size used on each client's local training
- `--local-epochs`: The number of epochs used on each client's local training
- `--epochs`: The number of global epochs
- `--gpu`: The id of the GPU to train on (if GPU is available) 
