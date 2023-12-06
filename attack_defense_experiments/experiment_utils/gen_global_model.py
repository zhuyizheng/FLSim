# uses code from https://github.com/dreamquark-ai/tabnet

import torch
from torchvision.models import resnet18
from flsim.utils.example_utils import FLModel
from flsim.utils.example_utils import SimpleConvNet
from pytorch_tabnet.augmentations import ClassificationSMOTE
from pytorch_tabnet.tab_model import TabNetClassifier


def gen_global_model(args, extra_info, USE_CUDA=True):
    cuda_enabled = torch.cuda.is_available() and USE_CUDA
    device = torch.device(f"cuda:{args.gpu}" if cuda_enabled else "cpu")

    if args.dataset == 'organamnist':
        model = SimpleConvNet(in_channels=1, num_classes=11)
        global_model = FLModel(model, device)
        if cuda_enabled:
            global_model.fl_cuda()
    elif args.dataset == 'organsmnist':
        model = resnet18()
        model.fc = torch.nn.Linear(512, 11)
        global_model = FLModel(model, device)
        if cuda_enabled:
            global_model.fl_cuda()
    elif args.dataset == 'forest':
        aug = ClassificationSMOTE(p=0.2, device_name=device)
        clf = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            cat_idxs=extra_info['cat_idxs'],
            cat_dims=extra_info['cat_dims'],
            cat_emb_dim=1,
            lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"gamma": 0.95,
                              "step_size": 20},
            scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15,
            device_name=f"cuda:{args.gpu}" if cuda_enabled else "cpu"
        )

        from flsim.utils.simple_batch_metrics import FLBatchMetrics

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
                    device=torch.device(f"cuda:{args.gpu}" if cuda_enabled else "cpu")
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

        global_model = TabNetFLModel(clf)
        global_model.construct_network(X_train=extra_info['X_train'], y_train=extra_info['y_train'],
                                       eval_set=[(extra_info['X_train'], extra_info['y_train']), (extra_info['X_test'], extra_info['y_test'])],
                                       eval_name=['train', 'test'],
                                       max_epochs=args.local_epochs, patience=100,
                                       batch_size=args.local_batch_size, virtual_batch_size=256,
                                       augmentations=aug,
                                       device=device)
    else:
        raise ValueError("Wrong dataset!!")

    return global_model
