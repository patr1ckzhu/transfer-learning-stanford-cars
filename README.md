# Transfer Learning on Stanford Cars

Fine-tune ImageNet-pretrained models on [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) (196 car classes). Achieves **92.99% test accuracy** with EfficientNet-B4, surpassing the published baseline (~90%).

## Results

| Model | Config | Optimizer | Test Acc | Training Time |
|-------|--------|-----------|----------|---------------|
| ResNet-50 | 224px, batch=128 | SGD lr=0.1 | 89.38% | 18.9 min |
| ResNet-50 | 448px, batch=64 | SGD lr=0.05 | 91.78% | 50.3 min |
| EfficientNet-B4 | 380px, batch=64 | SGD lr=0.05 | 91.57% | 58.9 min |
| **EfficientNet-B4** | **380px, batch=64** | **AdamW lr=1e-3** | **92.99%** | **56.8 min** |

### GradCAM Visualizations

**Random test samples** -- the model focuses on car body shapes, grilles, and taillights for classification:

![GradCAM Random Samples](gradcam_samples.png)

**Worst-class error samples** -- on difficult cases, attention is often diffuse or on irrelevant regions (background, watermarks):

![GradCAM Worst Classes](gradcam_worst.png)

### Most Confused Pairs

The top errors are between visually near-identical cars (same model different year/trim, or rebadged vehicles):

| True Class | Predicted As | Errors |
|-----------|-------------|--------|
| Audi S5 Coupe 2012 | Audi A5 Coupe 2012 | 14 |
| Audi TTS Coupe 2012 | Audi TT Hatchback 2011 | 13 |
| Audi TT Hatchback 2011 | Audi TTS Coupe 2012 | 12 |
| Dodge Caliber Wagon 2007 | Dodge Caliber Wagon 2012 | 12 |
| Dodge Sprinter Cargo Van 2009 | Mercedes-Benz Sprinter Van 2012 | 12 |

## Training Configuration (Best: EfficientNet-B4)

```
Model:      EfficientNet-B4 (ImageNet pretrained)
Resolution: 380x380
Optimizer:  AdamW (lr=1e-3, weight_decay=1e-2)
Scheduler:  Cosine Annealing
Augment:    RandomResizedCrop(380) + RandomHorizontalFlip
Batch size: 64
Epochs:     90
Speedups:   AMP + torch.compile + channels_last
GPU Memory: ~14 GB / 16 GB (RTX 5080)
```

## Experiment Log

10 runs total. Runs 1-6 failed (41-52%) due to lr too low, frozen backbone, and over-regularization. Run 7 followed a baseline and succeeded. Runs 8-10 explored higher resolution and EfficientNet-B4.

| Run | Model | Strategy | lr | Resolution | Test Acc |
|-----|-------|----------|----|-----------|----------|
| 1 | ResNet-50 | 2-stage AdamW | bb:1e-4 fc:1e-3 | 224 | 50.8% |
| 2 | ResNet-50 | 2-stage AdamW | bb:1e-4 fc:1e-3 | 224 | 52.2% |
| 3 | ResNet-50 | 2-stage AdamW | bb:1e-3 fc:1e-2 | 448 | 49.7% |
| 4 | ResNet-50 | 2-stage AdamW | bb:3e-4 fc:3e-3 | 448 | 50.3% |
| 5 | ResNet-50 | 2-stage SGD | bb:1e-3 fc:1e-2 | 224 | 51.4% |
| 6 | ResNet-50 | 2-stage SGD | bb:5e-4 fc:5e-3 | 448 | 41.2% |
| 7 | ResNet-50 | 1-stage SGD | 0.1 | 224 | 89.38% |
| 8 | ResNet-50 | 1-stage SGD | 0.05 | 448 | 91.78% |
| 9 | EffNet-B4 | 1-stage SGD | 0.05 | 380 | 91.57% |
| **10** | **EffNet-B4** | **1-stage AdamW+cosine** | **1e-3** | **380** | **92.99%** |

## Lessons Learned

1. **Learning rate is king** -- baseline uses lr=0.1; our failed runs maxed at 0.01 (10x too low). Fine-tuning needs aggressive lr to adapt backbone features from coarse ImageNet classes to fine-grained car models.

2. **Don't freeze the backbone** -- single-stage full-parameter training with a unified lr outperforms 2-stage frozen+unfrozen approaches. SGD gradients naturally scale per-layer updates.

3. **Higher resolution helps, but only with correct lr** -- Run 6 used 448px with lr=5e-4 and got 41%. Run 8 used 448px with lr=0.05 and got 91.78%. The resolution wasn't the problem -- the lr was.

4. **Batch size and lr are coupled** -- batch=128 + lr=0.1 is a validated combo. Small batches need proportionally lower lr.

5. **Reproduce the baseline first** -- don't try "advanced" techniques before confirming the simplest approach works.

6. **Architecture and optimizer must match** -- EfficientNet-B4 with SGD got 91.57%, but with AdamW + cosine it jumped to 92.99%. ResNet works best with SGD + high lr; EfficientNet works best with AdamW + low lr.

## Usage

### Training

```bash
# EfficientNet-B4 (best accuracy, 92.99%)
python train.py --arch efficientnet_b4 --resolution 380 --batch-size 64 \
    --lr 1e-3 --optimizer adamw --weight-decay 1e-2 --scheduler cosine

# ResNet-50 448px (91.78%)
python train.py --resolution 448 --batch-size 64 --lr 0.05

# ResNet-50 224px (fastest, 89.38%)
python train.py --resolution 224 --batch-size 128 --lr 0.1
```

Saves `best_model.pth` (best test accuracy checkpoint).

### Evaluation + Analysis

```bash
pip install grad-cam matplotlib  # one-time

python eval.py --arch efficientnet_b4 --resolution 380       # EfficientNet-B4
python eval.py --arch efficientnet_b4 --resolution 380 --tta  # with TTA
python eval.py --resolution 448                              # ResNet-50 448px
python eval.py --num-gradcam 32                              # more GradCAM samples
```

Outputs:
- Per-class accuracy with class names
- Top-10 most confused class pairs
- `gradcam_samples.png` -- GradCAM on random test samples
- `gradcam_worst.png` -- GradCAM on misclassified samples from worst classes
- With `--tta`: accuracy boost via 3-scale x 2-flip averaging (+0.15% at 448px, +0.95% at 224px)

## Dataset

[Stanford Cars](https://huggingface.co/datasets/tanganke/stanford_cars) via HuggingFace Datasets:
- 196 classes (make + model + year, e.g. "BMW M3 Coupe 2012")
- 8,144 train / 8,041 test images

## Dependencies

- Python 3.11
- PyTorch 2.10, torchvision 0.25
- `datasets` (HuggingFace)
- `grad-cam`, `matplotlib` (for eval.py)
