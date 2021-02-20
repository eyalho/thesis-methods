import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
# from pl_bolts.models.self_supervised import SimCLR, CPCV2

from torchvision import transforms as transforms


# MNISTDataModule doesn't work!! it need some work to modify it... :(

# SimCLR Transform
class ModifiedSimCLRTrainDataTransform(object):
    """
    Transforms for SimCLR
        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()

    Example::
        transform = ModifiedSimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(self, input_height: int = 224, normalize=None) -> None:

        self.input_height = input_height
        self.normalize = normalize

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        data_transforms = transforms.Compose(data_transforms)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.train_transform = transforms.Compose([data_transforms, self.final_transform])

        # add online train transform of the size of global view
        self.online_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.input_height),
            transforms.RandomHorizontalFlip(), self.final_transform
        ])

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj, self.online_transform(sample)


#######################################################################################
import math
from argparse import ArgumentParser
from typing import Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from pl_bolts.models.self_supervised.resnets import resnet18, resnet50, ResNet, MODEL_URLS, BasicBlock
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)


def resnet18_mnist(pretrained: bool = False, progress: bool = True, **kwargs):
    def _resnet(arch, block, layers, pretrained, progress, **kwargs):
        model = ResNet(block, layers, **kwargs)
        if pretrained:
            from torch.hub import load_state_dict_from_url
            state_dict = load_state_dict_from_url(MODEL_URLS[arch], progress=progress)
            model.load_state_dict(state_dict)
        return model

    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model


class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        return grad_input[torch.distributed.get_rank() * ctx.batch_size:(torch.distributed.get_rank() + 1) *
                                                                        ctx.batch_size]


class Projection(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):

    def __init__(
            self,
            gpus: int,
            num_samples: int,
            batch_size: int,
            dataset: str,
            num_nodes: int = 1,
            arch: str = 'resnet18_mnist',
            hidden_mlp: int = 512,
            feat_dim: int = 128,
            warmup_epochs: int = 10,
            max_epochs: int = 100,
            temperature: float = 0.1,
            first_conv: bool = True,
            maxpool1: bool = True,
            optimizer: str = 'adam',
            lars_wrapper: bool = True,
            exclude_bn_bias: bool = False,
            start_lr: float = 0.,
            learning_rate: float = 1e-3,
            final_lr: float = 0.,
            weight_decay: float = 1e-6,
            **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.lars_wrapper = lars_wrapper
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.encoder = self.init_model()

        self.projection = Projection(input_dim=self.hidden_mlp, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)

        # compute iters per epoch
        nb_gpus = len(self.gpus) if isinstance(gpus, (list, tuple)) else self.gpus
        assert isinstance(nb_gpus, int)
        global_batch_size = self.num_nodes * nb_gpus * self.batch_size if nb_gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        # define LR schedule
        warmup_lr_schedule = np.linspace(
            self.start_lr, self.learning_rate, self.train_iters_per_epoch * self.warmup_epochs
        )
        iters = np.arange(self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs))
        cosine_lr_schedule = np.array([
            self.final_lr + 0.5 * (self.learning_rate - self.final_lr) *
            (1 + math.cos(math.pi * t / (self.train_iters_per_epoch * (self.max_epochs - self.warmup_epochs))))
            for t in iters
        ])

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    def init_model(self):
        if self.arch == 'resnet18':
            backbone = resnet18
        elif self.arch == 'resnet50':
            backbone = resnet50
        elif self.arch == 'resnet18_mnist':
            backbone = resnet18_mnist

        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

    def forward(self, x):
        # bolts resnet returns a list
        return self.encoder(x)[-1]

    def shared_step(self, batch):
        if self.dataset == "mnist":
            pass
            # print("")
            # print("batch_data1:", len(batch))
            # print("batch_data1.2:", batch[1])
            # print("batch_data2:", len(batch[0]))
            # print("batch_data3:", len(batch[0][0]))
            # print("batch_data3.1:", len(batch[0][1]))
            # print("batch_data3.2:", batch[0][2].shape)
            # print("batch_data4:", len(batch[0][0].shape))

        if self.dataset == 'stl10':
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        # final image in tuple is for online eval
        (img1, img2, _), y = batch

        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.temperature)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        # log LR (LearningRateLogger callback doesn't work with LARSWrapper)
        self.log('learning_rate', self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)

        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{
            'params': params,
            'weight_decay': weight_decay
        }, {
            'params': excluded_params,
            'weight_decay': 0.,
        }]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.lars_wrapper:
            optimizer = LARSWrapper(
                optimizer,
                eta=0.001,  # trust coefficient
                clip=False
            )

        return optimizer

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer: Optimizer = None,
            optimizer_idx: int = None,
            optimizer_closure: Optional[Callable] = None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[self.trainer.global_step]

        # from lightning
        if not isinstance(optimizer, LightningOptimizer):
            # wraps into LightingOptimizer only for running step
            optimizer = LightningOptimizer.to_lightning_optimizer(optimizer, self.trainer)
        optimizer.step(closure=optimizer_closure)

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss


#######################################################################################
ds_name = "mnist"

if ds_name == "cifar10":
    # data
    dm = CIFAR10DataModule(num_workers=0)

    input_height = dm.dims[-1]
    print("input_height", input_height)
    dm.train_transforms = ModifiedSimCLRTrainDataTransform(input_height)
    dm.val_transforms = ModifiedSimCLRTrainDataTransform(input_height)

    model = SimCLR(num_samples=dm.num_samples, batch_size=dm.batch_size, dataset='cifar10', gpus=0)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, datamodule=dm)

if ds_name == "mnist":
    # In order to train it I change some stuff:
    # - arch: str = 'resnet18_mnist', and a resent with 1-channel input layer was supplied
    # - hidden_mlp: int = 512, to fix projection (found value by debugging error mat1,mat2 dimension were wrong..)

    # data
    dm = MNISTDataModule()

    input_height = dm.dims[-1]
    print("input_height", input_height)
    dm.train_transforms = ModifiedSimCLRTrainDataTransform(input_height)
    dm.val_transforms = ModifiedSimCLRTrainDataTransform(input_height)

    model = SimCLR(num_samples=50000, batch_size=dm.batch_size, arch="resnet18_mnist", hidden_mlp=512, dataset='mnist',
                   gpus=0)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, datamodule=dm)
