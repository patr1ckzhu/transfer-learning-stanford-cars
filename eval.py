"""Stanford Cars 评估脚本 — overall + per-class accuracy."""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from datasets import load_dataset

from train import StanfordCarsDataset, IMAGENET_MEAN, IMAGENET_STD


def build_model(num_classes=196):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stanford Cars model")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform_eval = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("Loading Stanford Cars test set...")
    raw = load_dataset("tanganke/stanford_cars")
    test_set = StanfordCarsDataset(raw["test"], transform_eval)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"Test samples: {len(test_set)}")

    num_classes = 196
    model = build_model(num_classes)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # 统计 overall + per-class
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            for c in range(num_classes):
                mask = targets == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (preds[mask] == c).sum().item()

    overall_acc = 100.0 * class_correct.sum() / class_total.sum()
    print(f"\nOverall test accuracy: {overall_acc:.2f}%")

    print(f"\nPer-class accuracy ({num_classes} classes):")
    print("-" * 40)
    for c in range(num_classes):
        if class_total[c] > 0:
            acc = 100.0 * class_correct[c] / class_total[c]
            print(f"  Class {c:3d}: {acc:6.2f}%  ({int(class_correct[c])}/{int(class_total[c])})")

    # 统计 worst / best classes
    valid_mask = class_total > 0
    class_acc = torch.zeros(num_classes)
    class_acc[valid_mask] = class_correct[valid_mask] / class_total[valid_mask]

    sorted_idx = class_acc.argsort()
    print(f"\nBottom 5 classes:")
    for i in range(5):
        c = sorted_idx[i].item()
        print(f"  Class {c:3d}: {100*class_acc[c]:.2f}%")
    print(f"\nTop 5 classes:")
    for i in range(5):
        c = sorted_idx[-(i+1)].item()
        print(f"  Class {c:3d}: {100*class_acc[c]:.2f}%")


if __name__ == "__main__":
    main()
