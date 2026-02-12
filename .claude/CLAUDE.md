# Transfer Learning: ResNet-50 + Stanford Cars

## 项目目标
用 ImageNet 预训练 ResNet-50 fine-tune Stanford Cars（196 类车型），学习迁移学习。

## 环境

### Mac（开发机）
- 包管理: uv, Python 3.11
- torch 2.10.0, torchvision 0.25.0 (CPU)
- 代码路径: /Users/patrick/PycharmProjects/deep-learning/cv/transfer-stanford-cars

### Ubuntu（训练机）
- GPU: NVIDIA GeForce RTX 5080 16GB
- conda env: `cv` (Python 3.11)
- torch 2.10.0+cu130, torchvision 0.25.0+cu130
- 额外依赖: `datasets` (HuggingFace)
- 代码路径: /home/patrick/PycharmProjects/transfer-stanford-cars
- HuggingFace token 已配置（`huggingface-cli login`）

### SSH
- `ssh ubuntu` → 192.168.1.224
- `scp <files> ubuntu:/home/patrick/PycharmProjects/transfer-stanford-cars/`

## 数据集
- `datasets.load_dataset("tanganke/stanford_cars")`
- 196 类（品牌+车型+年份，如 "BMW M3 Coupe 2012"）
- train 8144 / test 8041，列: `image` (PIL), `label` (0-195)
- 不再切分 val，全量 8144 训练，用 test set 做验证选 best model（与 baseline 一致）
- 图片分辨率: avg 608x403, min 85x62, max 2405x1600
- 每类样本数: min 24, max 68, avg 41.6
- 数据已验证: 预训练 ResNet-50 正确识别所有车图为 ImageNet 车类（jeep/sports car/minivan 等）

## 项目结构
```
train.py  — 单阶段全参数 fine-tune（参考 baseline，SGD lr=0.1 + step decay）
eval.py   — 加载 best_model.pth 评估 overall + per-class accuracy
```

## 当前训练配置（第 7 轮 — 参考 baseline）
- 分辨率: 224（匹配 ImageNet 预训练）
- 单阶段全参数训练，无冻结，无 differential lr
- 优化器: SGD (lr=0.1, momentum=0.9, weight_decay=1e-4)
- LR 调度: step decay ÷10 every 30 epochs（epoch 0-29: 0.1, 30-59: 0.01, 60-89: 0.001）
- 数据增强: RandomResizedCrop(224) + RandomHorizontalFlip（最简增强）
- 无 Dropout, 无 label_smoothing, 无 ColorJitter
- AMP, torch.compile, channels_last
- batch_size=128
- epochs=90

## 实验记录

| Run | 策略 | 分辨率 | lr | wd | batch | 增强 | epochs | Test% | 备注 |
|-----|------|--------|-----|-----|-------|------|--------|-------|------|
| 1 | 两阶段+AdamW | 224 | bb1e-4/fc1e-3 | 5e-2 | 16 | crop+flip | 20 | 50.8 | wd 太重+太少 epoch |
| 2 | 两阶段+AdamW | 224 | bb1e-4/fc1e-3 | 5e-2 | 16 | 重度 | 50 | 52.2 | 过度正则→underfitting |
| 3 | 两阶段+AdamW | 448 | bb1e-3/fc1e-2 | 5e-2 | 16 | 重度 | 50 | 49.7 | lr 太高炸模型 |
| 4 | 两阶段+AdamW | 448 | bb3e-4/fc3e-3 | 5e-2 | 16 | 重度 | 50 | 50.3 | wd 太重+过度正则 |
| 5 | 两阶段+SGD | 224 | bb1e-3/fc1e-2 | 1e-4 | 16 | crop+flip | 100 | 51.4 | 过拟合(train73%/val52%) |
| 6 | 两阶段+SGD | 448 | bb5e-4/fc5e-3 | 1e-4 | 16 | 适度CJ | 100 | 41.2 | 448 更差，Phase1仅19% |
| **7** | **单阶段SGD** | **224** | **0.1统一** | **1e-4** | **128** | **crop+flip** | **90** | **89.38%** | **参考baseline，成功** |

### Run 7 训练曲线
- Epoch 1: train 10.6% / test 19.6%（高 lr 快速启动）
- Epoch 10: train 83.0% / test 73.2%
- Epoch 30: train 91.0% / test 81.0%（lr=0.1 阶段结束）
- Epoch 31: lr 降至 0.01 → test 立刻跳到 86.9%（step decay 效果显著）
- Epoch 44: test 88.9%
- Epoch 57: test 89.3%
- Epoch 61: lr 降至 0.001 → 精细调整
- Epoch 89: **best test 89.38%**
- 总训练时间: 18.9 min（RTX 5080）

### 关键教训（Run 1-6 为什么失败）
1. **LR 太低是最致命的错误**: baseline 用 0.1，我们最高只用 1e-2（低 10 倍），backbone lr 只有 5e-4（低 200 倍）。Fine-tune 需要足够大的 lr 才能让 backbone 适应新的细粒度分类任务
2. **两阶段冻结策略是过度设计**: 冻结 backbone 限制了模型适应能力，differential lr 增加了调参复杂度但没有带来收益。直接全参数训练 + 统一 lr，让 SGD 梯度自然决定每层更新幅度
3. **分辨率必须匹配预训练**: 448 输入破坏了 ResNet-50 在 224 上学到的特征表示
4. **batch size 和 lr 是绑定的**: batch=128 + lr=0.1 是经过验证的组合，小 batch 需要按比例降 lr
5. **简单方法优先**: 最朴素的配置（SGD + step decay + crop+flip）反而最有效

## 参考
- 论文基准: ResNet-50 + Stanford Cars ≈ 85-90% test accuracy
- Baseline 项目: `PyTorch-Stanford-Cars-Baselines`（ResNet-50 pretrained = 90.0%）
- 我们的最佳结果: **89.38%**（与 baseline 基本持平）
- 用户有 ResNet-18 CIFAR-10 从零实现经验
