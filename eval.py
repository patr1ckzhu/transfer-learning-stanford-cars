"""Stanford Cars 评估脚本 — overall + per-class accuracy + confusion analysis + GradCAM."""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from datasets import load_dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import StanfordCarsDataset, IMAGENET_MEAN, IMAGENET_STD


def build_model(num_classes=196):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def denormalize(tensor):
    """Convert normalized tensor back to [0, 1] numpy RGB image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def generate_gradcam_grid(model, dataset, indices, class_names, target_layer,
                          device, save_path, title="GradCAM Samples"):
    """Generate a GradCAM visualization grid for given sample indices."""
    n = len(indices)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    cam = GradCAM(model=model, target_layers=[target_layer])

    for ax_idx, sample_idx in enumerate(indices):
        img_tensor, label = dataset[sample_idx]
        input_batch = img_tensor.unsqueeze(0).to(device)

        # Forward to get prediction
        with torch.no_grad():
            output = model(input_batch)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            conf = probs[0, pred].item()

        # GradCAM (use predicted class as target)
        grayscale_cam = cam(input_tensor=input_batch)[0]
        rgb_img = denormalize(img_tensor)
        vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        ax = axes[ax_idx]
        ax.imshow(vis)
        correct = pred == label
        mark = "O" if correct else "X"
        pred_name = class_names[pred]
        # Truncate long names
        if len(pred_name) > 30:
            pred_name = pred_name[:27] + "..."
        color = "green" if correct else "red"
        ax.set_title(f"[{mark}] {pred_name}\nconf={conf:.2f}", fontsize=8, color=color)
        ax.axis("off")

    # Hide unused axes
    for ax_idx in range(n, len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stanford Cars model")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-gradcam", type=int, default=16,
                        help="Number of random GradCAM samples")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (3 scales x flip = 6 views)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("Loading Stanford Cars test set...")
    raw = load_dataset("tanganke/stanford_cars")
    class_names = raw["train"].features["label"].names  # 196 class names
    test_set = StanfordCarsDataset(raw["test"], transform_eval)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"Test samples: {len(test_set)}")

    # --- Model ---
    num_classes = 196
    model = build_model(num_classes)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # Strip _orig_mod. prefix from torch.compile'd checkpoint
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ==================== 1. Overall + Per-class Accuracy ====================
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(targets.cpu())
            for c in range(num_classes):
                mask = targets == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (preds[mask] == c).sum().item()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Build confusion matrix
    for p, t in zip(all_preds, all_labels):
        confusion[t, p] += 1

    overall_acc = 100.0 * class_correct.sum() / class_total.sum()
    print(f"\nOverall test accuracy: {overall_acc:.2f}%")

    # Per-class accuracy
    valid_mask = class_total > 0
    class_acc = torch.zeros(num_classes)
    class_acc[valid_mask] = class_correct[valid_mask] / class_total[valid_mask]

    print(f"\nPer-class accuracy ({num_classes} classes):")
    print("-" * 70)
    for c in range(num_classes):
        if class_total[c] > 0:
            acc = 100.0 * class_acc[c]
            print(f"  {c:3d} {class_names[c]:45s} {acc:6.2f}%  "
                  f"({int(class_correct[c])}/{int(class_total[c])})")

    # Bottom / Top 10
    sorted_idx = class_acc.argsort()
    print(f"\nBottom 10 classes:")
    for i in range(10):
        c = sorted_idx[i].item()
        print(f"  {c:3d} {class_names[c]:45s} {100*class_acc[c]:.2f}%")
    print(f"\nTop 10 classes:")
    for i in range(10):
        c = sorted_idx[-(i + 1)].item()
        print(f"  {c:3d} {class_names[c]:45s} {100*class_acc[c]:.2f}%")

    # ==================== TTA (optional) ====================
    if args.tta:
        print(f"\n{'='*60}")
        print("Running TTA (3 scales x 2 flips = 6 views)...")
        tta_transforms = []
        for size in [224, 256, 288]:
            for flip in [False, True]:
                t = [transforms.Resize(size), transforms.CenterCrop(224)]
                if flip:
                    t.append(transforms.RandomHorizontalFlip(p=1.0))
                t.extend([transforms.ToTensor(),
                          transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
                tta_transforms.append(transforms.Compose(t))

        all_logits = []
        for t_idx, tfm in enumerate(tta_transforms):
            ds = StanfordCarsDataset(raw["test"], tfm)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
            logits = []
            with torch.no_grad():
                for inputs, _ in loader:
                    logits.append(model(inputs.to(device)).cpu())
            all_logits.append(torch.cat(logits))
            print(f"  Pass {t_idx+1}/6 done")

        avg_logits = torch.stack(all_logits).mean(dim=0)
        tta_preds = avg_logits.argmax(dim=1)

        # TTA overall accuracy
        tta_correct = (tta_preds == all_labels).sum().item()
        tta_acc = 100.0 * tta_correct / len(all_labels)
        print(f"\nTTA accuracy: {tta_acc:.2f}%  (baseline: {overall_acc:.2f}%,"
              f" delta: {tta_acc - overall_acc:+.2f}%)")

        # TTA per-class accuracy comparison for bottom 10
        tta_class_correct = torch.zeros(num_classes)
        for i in range(len(all_labels)):
            t = all_labels[i].item()
            if tta_preds[i].item() == t:
                tta_class_correct[t] += 1
        tta_class_acc = torch.zeros(num_classes)
        tta_class_acc[valid_mask] = tta_class_correct[valid_mask] / class_total[valid_mask]

        print(f"\nBottom 10 comparison (baseline → TTA):")
        for i in range(10):
            c = sorted_idx[i].item()
            old = 100 * class_acc[c]
            new = 100 * tta_class_acc[c]
            delta = new - old
            print(f"  {c:3d} {class_names[c]:45s} {old:.1f}% → {new:.1f}% ({delta:+.1f}%)")
        print(f"{'='*60}")

    # ==================== 2. Most Confused Pairs ====================
    # Zero out diagonal, find top-10 off-diagonal entries
    conf_off = confusion.clone().float()
    conf_off.fill_diagonal_(0)
    print(f"\nTop 10 most confused pairs (true → predicted, count):")
    print("-" * 80)
    for _ in range(10):
        idx = conf_off.argmax().item()
        true_c = idx // num_classes
        pred_c = idx % num_classes
        count = int(conf_off[true_c, pred_c])
        if count == 0:
            break
        true_name = class_names[true_c]
        pred_name = class_names[pred_c]
        print(f"  {true_name:40s} → {pred_name:40s} ({count})")
        conf_off[true_c, pred_c] = 0  # zero out to find next

    # ==================== 3. GradCAM — Random Samples ====================
    if args.num_gradcam <= 0:
        print("\nSkipping GradCAM (--num-gradcam 0).")
        return

    print(f"\nGenerating GradCAM for {args.num_gradcam} random test samples...")
    target_layer = model.layer4[-1]
    random_indices = random.sample(range(len(test_set)), args.num_gradcam)
    generate_gradcam_grid(model, test_set, random_indices, class_names,
                          target_layer, device, "gradcam_samples.png",
                          title="GradCAM — Random Test Samples")

    # ==================== 4. GradCAM — Worst Classes (错误样本) ====================
    print("Generating GradCAM for worst-class error samples...")
    worst_indices = []
    bottom5 = [sorted_idx[i].item() for i in range(5)]
    for c in bottom5:
        # Find misclassified samples for this class
        error_idxs = []
        for i in range(len(all_labels)):
            if all_labels[i].item() == c and all_preds[i].item() != c:
                error_idxs.append(i)
        if error_idxs:
            worst_indices.append(random.choice(error_idxs))

    if worst_indices:
        generate_gradcam_grid(model, test_set, worst_indices, class_names,
                              target_layer, device, "gradcam_worst.png",
                              title="GradCAM — Worst-class Error Samples")
    else:
        print("No error samples found for bottom-5 classes (all correct).")


if __name__ == "__main__":
    main()
