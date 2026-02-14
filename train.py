"""Stanford Cars 迁移学习训练脚本.

支持 ResNet-50 和 EfficientNet-B4, 单阶段全参数 fine-tune, SGD + step decay.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, EfficientNet_B4_Weights

from datasets import load_dataset


class StanfordCarsDataset(Dataset):
    """HuggingFace Stanford Cars → PyTorch Dataset wrapper."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_dataloaders(batch_size, resolution=224):
    resize_eval = int(resolution * 256 / 224)  # 224→256, 448→512
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(resize_eval),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("Loading Stanford Cars from HuggingFace...")
    raw = load_dataset("tanganke/stanford_cars")
    raw_train = raw["train"]  # 8144
    raw_test = raw["test"]    # 8041

    train_set = StanfordCarsDataset(raw_train, transform_train)
    test_set = StanfordCarsDataset(raw_test, transform_eval)

    loader_kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def build_model(arch="resnet50", num_classes=196):
    if arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, 100.0 * correct / total


def adjust_learning_rate(optimizer, epoch, initial_lr, step_size=30):
    """每 step_size 个 epoch 将 lr 衰减 10 倍."""
    lr = initial_lr * (0.1 ** (epoch // step_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning: ResNet-50 → Stanford Cars")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adamw"])
    parser.add_argument("--scheduler", type=str, default="step",
                        choices=["step", "cosine"])
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--arch", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b4"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(args.batch_size, args.resolution)
    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    model = build_model(arch=args.arch, num_classes=196)
    model = model.to(device, memory_format=torch.channels_last)
    if device.type == "cuda":
        model = torch.compile(model)
        print("torch.compile enabled")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        )

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    print(f"\nConfig: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}, "
          f"resolution={args.resolution}, optimizer={args.optimizer}, scheduler={args.scheduler}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,}")

    best_test_acc = 0.0
    train_start = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()
        if scheduler:
            lr = optimizer.param_groups[0]['lr']
        else:
            lr = adjust_learning_rate(optimizer, epoch, args.lr, args.step_size)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        if scheduler:
            scheduler.step()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        total_elapsed = time.time() - train_start
        avg_epoch_time = total_elapsed / (epoch + 1)
        eta = avg_epoch_time * (args.epochs - epoch - 1)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"lr {lr:.4f} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc:.2f}% | "
            f"{elapsed:.1f}s | ETA {eta/60:.1f}min"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  → saved best model (test acc {best_test_acc:.2f}%)")

    total_time = time.time() - train_start
    print(f"\nTotal training time: {total_time/60:.1f} min")
    print(f"Best test accuracy: {best_test_acc:.2f}%")


if __name__ == "__main__":
    main()
