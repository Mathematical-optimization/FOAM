import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import math
from typing import Type, Any, Callable, Union, List, Optional

# Shampoo imports
from optimizers.distributed_shampoo.distributed_shampoo import DistributedShampoo
from optimizers.distributed_shampoo.shampoo_types import (
    AdamGraftingConfig,
    DDPShampooConfig,
    CommunicationDType
)

# ==========================================
# 1. Setup & Utility
# ==========================================
def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

# ==========================================
# 2. Ghost Batch Normalization
# ==========================================
class GhostBatchNorm2d(nn.Module):
    def __init__(self, num_features, virtual_batch_size=64, momentum=0.1, eps=1e-5):
        super(GhostBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm2d(num_features, momentum=momentum, eps=eps)

    def forward(self, x):
        if not self.training or x.size(0) <= self.virtual_batch_size:
            return self.bn(x)

        chunks = x.chunk(int(torch.ceil(torch.tensor(x.size(0) / self.virtual_batch_size))), 0)
        res = [self.bn(chunk) for chunk in chunks]
        return torch.cat(res, dim=0)

# ==========================================
# 3. ResNet Model Architecture
# ==========================================
class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None, virtual_batch_size=64):
        super(ResNet, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, GhostBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_algoperf_resnet50(virtual_batch_size=64):
    norm_layer = lambda num_features: GhostBatchNorm2d(
        num_features, virtual_batch_size=virtual_batch_size
    )
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        zero_init_residual=True,
        norm_layer=norm_layer,
        virtual_batch_size=virtual_batch_size
    )

# ==========================================
# 4. Dataset & Loader
# ==========================================
def get_dataloaders(data_path, batch_size, workers, world_size, global_rank):
    traindir = os.path.join(data_path, 'train')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=train_sampler
    )
    
    return train_loader

# ==========================================
# 5. Main
# ==========================================
def main(args):
    local_rank = setup()
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if global_rank == 0:
        print(f"Starting ResNet-50 training with Distributed Shampoo")
        print(f"World Size: {world_size}, Batch Size per GPU: {args.batch_size}")

    # Create Model
    model = create_algoperf_resnet50(virtual_batch_size=args.batch_size).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Distributed Shampoo Optimizer
    distributed_config = DDPShampooConfig(
        communication_dtype=CommunicationDType.FP32,
        num_trainers_per_group=world_size,
        communicate_params=False
    )

    optimizer = DistributedShampoo(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        epsilon=1e-8,
        weight_decay=args.weight_decay,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        use_decoupled_weight_decay=True,
        grafting_config=AdamGraftingConfig(beta2=args.beta2, epsilon=1e-8),
        distributed_config=distributed_config,
        use_protected_eigh=True
    )

    criterion = nn.CrossEntropyLoss().to(local_rank)
    train_loader = get_dataloaders(args.data_path, args.batch_size, args.workers, world_size, global_rank)

    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        
        for i, (images, target) in enumerate(train_loader):
            images = images.to(local_rank, non_blocking=True)
            target = target.to(local_rank, non_blocking=True)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % args.log_interval == 0 and global_rank == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{i}] Loss: {loss.item():.4f}")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ResNet50 ImageNet Training with Shampoo')
    parser.add_argument('--data-path', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    
    # Shampoo Params
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--max-preconditioner-dim', type=int, default=1024)
    parser.add_argument('--precondition-frequency', type=int, default=100)
    parser.add_argument('--start-preconditioning-step', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=100)

    args = parser.parse_args()
    main(args)
