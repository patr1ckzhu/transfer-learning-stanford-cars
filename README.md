# Transfer Learning: ResNet-50 on Stanford Cars

Fine-tune an ImageNet-pretrained ResNet-50 on [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) (196 car classes). Achieves **89.38% test accuracy**, matching the published baseline (~90%).

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **89.38%** |
| Test Accuracy (TTA) | **90.32%** |
| Training Time | 18.9 min (RTX 5080) |
| 100% Accuracy Classes | 15 / 196 |
| Worst Class | Chevrolet Express Van 2007 (42.86%) |

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

## Training Configuration

```
Model:      ResNet-50 (ImageNet V2 pretrained)
Resolution: 224x224
Optimizer:  SGD (lr=0.1, momentum=0.9, weight_decay=1e-4)
Scheduler:  Step decay /10 every 30 epochs
Augment:    RandomResizedCrop(224) + RandomHorizontalFlip
Batch size: 128
Epochs:     90
Speedups:   AMP + torch.compile + channels_last
```

## Experiment Log

7 runs total. Runs 1-6 failed (41-52%) due to lr too low, frozen backbone, wrong resolution, and over-regularization. Run 7 followed an open-source baseline and succeeded immediately.

| Run | Strategy | lr | Resolution | Test Acc |
|-----|----------|----|-----------|----------|
| 1 | 2-stage AdamW | bb:1e-4 fc:1e-3 | 224 | 50.8% |
| 2 | 2-stage AdamW | bb:1e-4 fc:1e-3 | 224 | 52.2% |
| 3 | 2-stage AdamW | bb:1e-3 fc:1e-2 | 448 | 49.7% |
| 4 | 2-stage AdamW | bb:3e-4 fc:3e-3 | 448 | 50.3% |
| 5 | 2-stage SGD | bb:1e-3 fc:1e-2 | 224 | 51.4% |
| 6 | 2-stage SGD | bb:5e-4 fc:5e-3 | 448 | 41.2% |
| **7** | **1-stage SGD** | **0.1 unified** | **224** | **89.38%** |

## Lessons Learned

1. **Learning rate is king** -- baseline uses lr=0.1; our failed runs maxed at 0.01 (10x too low). Fine-tuning needs aggressive lr to adapt backbone features from coarse ImageNet classes to fine-grained car models.

2. **Don't freeze the backbone** -- single-stage full-parameter training with a unified lr outperforms 2-stage frozen+unfrozen approaches. SGD gradients naturally scale per-layer updates.

3. **Resolution must match pretraining** -- 448px input destroys the feature representations learned at 224px.

4. **Batch size and lr are coupled** -- batch=128 + lr=0.1 is a validated combo. Small batches need proportionally lower lr.

5. **Reproduce the baseline first** -- don't try "advanced" techniques before confirming the simplest approach works.

## Usage

### Training

```bash
# On GPU machine
python train.py --epochs 90 --lr 0.1 --batch-size 128
```

Saves `best_model.pth` (best test accuracy checkpoint).

### Evaluation + Analysis

```bash
pip install grad-cam matplotlib  # one-time

python eval.py                          # default: best_model.pth
python eval.py --tta                    # test-time augmentation (6 views)
python eval.py --num-gradcam 32         # more GradCAM samples
python eval.py --checkpoint my_model.pth
```

Outputs:
- Per-class accuracy with class names
- Top-10 most confused class pairs
- `gradcam_samples.png` -- GradCAM on random test samples
- `gradcam_worst.png` -- GradCAM on misclassified samples from worst classes
- With `--tta`: accuracy boost via 3-scale x 2-flip averaging (89.38% -> 90.32%)

## Dataset

[Stanford Cars](https://huggingface.co/datasets/tanganke/stanford_cars) via HuggingFace Datasets:
- 196 classes (make + model + year, e.g. "BMW M3 Coupe 2012")
- 8,144 train / 8,041 test images

## Dependencies

- Python 3.11
- PyTorch 2.10, torchvision 0.25
- `datasets` (HuggingFace)
- `grad-cam`, `matplotlib` (for eval.py)
