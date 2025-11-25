import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from typing import Type, Any, Callable, Union, List, Optional

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # Path to your ImageNet dataset
    # 이 경로에 'train'과 'val' 폴더가 있어야 합니다.
    data_path = "/path/to/imagenet_root" 
    
    # Training Params
    batch_size = 64         # 메모리 상황에 맞춰 조정 (논문은 대형 배치 사용)
    learning_rate = 1e-3    # 논문 베이스라인 설정 (AdamW 등)
    epochs = 90             # 일반적인 ImageNet 학습 Epoch
    num_classes = 1000      # ImageNet 클래스 개수
    
    # AlgoPerf Specifics
    virtual_batch_size = 64 # Ghost BN용 가상 배치 사이즈
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 8         # 데이터 로딩 속도를 위해 높게 설정

print(f"Using device: {Config.device}")

# ==========================================
# 2. Ghost Batch Normalization [cite: 1833-1837]
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
# 3. ResNet Model Architecture [cite: 1833]
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
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None):
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

        # AlgoPerf Requirement: Zero-initialize the last BN in each residual block [cite: 1837]
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

def create_algoperf_resnet50():
    norm_layer = lambda num_features: GhostBatchNorm2d(
        num_features, virtual_batch_size=Config.virtual_batch_size
    )
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=Config.num_classes,
        zero_init_residual=True, # AlgoPerf 필수 사항
        norm_layer=norm_layer    # AlgoPerf 필수 사항
    )

# ==========================================
# 4. ImageNet Dataset Loading [cite: 1827-1829]
# ==========================================
def get_dataloaders():
    traindir = os.path.join(Config.data_path, 'train')
    valdir = os.path.join(Config.data_path, 'val')
    
    # AlgoPerf Paper: "random crop and randomly flip the image" [cite: 1828]
    # 논문은 normalized to [0,1]이라고 명시했지만[cite: 1829],
    # ResNet 학습 안정성을 위해 통상적인 ImageNet mean/std 정규화를 포함하는 것이 일반적입니다.
    # 만약 논문의 [0,1]을 엄격히 따르려면 Normalize를 제외하면 됩니다.
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

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True,
        num_workers=Config.num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=Config.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

# ==========================================
# 5. Training Loop
# ==========================================
def train():
    if not os.path.exists(Config.data_path):
        print(f"Error: Data path '{Config.data_path}' does not exist.")
        print("Please download ImageNet and set 'Config.data_path' correctly.")
        return

    train_loader, val_loader = get_dataloaders()
    model = create_algoperf_resnet50().to(Config.device)
    
    # Optimizer (AlgoPerf baselines use AdamW, NadamW, etc.)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)
    
    print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Start Training on ImageNet...")

    for epoch in range(Config.epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, target) in enumerate(train_loader):
            images = images.to(Config.device, non_blocking=True)
            target = target.to(Config.device, non_blocking=True)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f"Epoch [{epoch+1}/{Config.epochs}] Batch [{i+1}] Loss: {running_loss/100:.4f}")
                running_loss = 0.0
                
        # Validation (Optional per epoch)
        # validate(val_loader, model, criterion)

    print("Training Finished.")

if __name__ == "__main__":
    train()
