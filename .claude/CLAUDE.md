# Transfer Learning: Stanford Cars

## 项目目标
用 ImageNet 预训练模型 fine-tune Stanford Cars（196 类车型），学习迁移学习。支持 ResNet-50 和 EfficientNet-B4。

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
train.py  — 单阶段全参数 fine-tune，支持 --arch resnet50/efficientnet_b4, --optimizer sgd/adamw, --scheduler step/cosine
eval.py   — 评估 + 分析：overall/per-class accuracy、混淆对分析、GradCAM 可视化
```

### eval.py 功能
1. **Overall + Per-class accuracy**: 196 类各自准确率，带类名显示（如 "BMW M3 Coupe 2012"）
2. **Bottom/Top 10**: 最差/最好的 10 个类
3. **混淆对分析**: 从 confusion matrix 提取 Top-10 最易混淆类别对（off-diagonal 最大值）
4. **TTA (Test-Time Augmentation)**: `--tta` 启用，3 尺度(224/256/288) x 2(原图+翻转) = 6 视角取平均
5. **GradCAM 可视化**:
   - `gradcam_samples.png` — 随机 16 张 test 样本热力图（`--num-gradcam` 控制数量，0 跳过）
   - `gradcam_worst.png` — Bottom-5 最差类各 1 张错误预测样本热力图
   - 目标层: ResNet-50 → `model.layer4[-1]`，EfficientNet-B4 → `model.features[-1]`
6. 自动处理 `torch.compile` checkpoint（strip `_orig_mod.` prefix）

### eval.py 依赖
- `grad-cam` (`pip install grad-cam`)
- `matplotlib`

## 当前最佳训练配置（Run 10 — EfficientNet-B4 + AdamW + Cosine）
- 模型: EfficientNet-B4（`--arch efficientnet_b4`）
- 分辨率: 380（EfficientNet-B4 预训练分辨率）
- 单阶段全参数训练
- 优化器: AdamW (lr=1e-3, weight_decay=1e-2)
- LR 调度: Cosine Annealing（`--scheduler cosine`）
- 数据增强: RandomResizedCrop(380) + RandomHorizontalFlip
- AMP, torch.compile, channels_last
- batch_size=64, epochs=90
- 显存: ~14 GB / 16 GB

## 实验记录

| Run | 策略 | 分辨率 | lr | wd | batch | 增强 | epochs | Test% | 备注 |
|-----|------|--------|-----|-----|-------|------|--------|-------|------|
| 1 | 两阶段+AdamW | 224 | bb1e-4/fc1e-3 | 5e-2 | 16 | crop+flip | 20 | 50.8 | wd 太重+太少 epoch |
| 2 | 两阶段+AdamW | 224 | bb1e-4/fc1e-3 | 5e-2 | 16 | 重度 | 50 | 52.2 | 过度正则→underfitting |
| 3 | 两阶段+AdamW | 448 | bb1e-3/fc1e-2 | 5e-2 | 16 | 重度 | 50 | 49.7 | lr 太高炸模型 |
| 4 | 两阶段+AdamW | 448 | bb3e-4/fc3e-3 | 5e-2 | 16 | 重度 | 50 | 50.3 | wd 太重+过度正则 |
| 5 | 两阶段+SGD | 224 | bb1e-3/fc1e-2 | 1e-4 | 16 | crop+flip | 100 | 51.4 | 过拟合(train73%/val52%) |
| 6 | 两阶段+SGD | 448 | bb5e-4/fc5e-3 | 1e-4 | 16 | 适度CJ | 100 | 41.2 | 448 更差，Phase1仅19% |
| 7 | 单阶段SGD | 224 | 0.1统一 | 1e-4 | 128 | crop+flip | 90 | 89.38% | 参考baseline，成功 |
| 8 | 单阶段SGD | 448 | 0.05统一 | 1e-4 | 64 | crop+flip | 90 | 91.78% | 高分辨率+linear scaling |
| 9 | EffNet-B4+SGD | 380 | 0.05统一 | 1e-4 | 64 | crop+flip | 90 | 91.57% | SGD 不适合 EffNet |
| **10** | **EffNet-B4+AdamW** | **380** | **1e-3** | **1e-2** | **64** | **crop+flip+cosine** | **90** | **92.99%** | **架构+优化器配套** |

### Run 10 训练曲线（EfficientNet-B4 + AdamW + Cosine）
- Epoch 1: train 11.2% / test 42.0%（AdamW 启动比 SGD 快）
- Epoch 8: test 88.7%
- Epoch 14: test 90.7%
- Epoch 48: test 92.1%（cosine 平滑爬升，无 step decay 跳跃）
- Epoch 71: **best test 92.99%**
- 总训练时间: 56.8 min（RTX 5080）

### Run 8 训练曲线（ResNet-50 448px）
- Epoch 31: lr 降至 0.005 → test 跳到 89.9%
- Epoch 81: **best test 91.78%**
- 总训练时间: 50.3 min

### Run 7 训练曲线（ResNet-50 224px）
- Epoch 31: lr 降至 0.01 → test 跳到 86.9%
- Epoch 89: **best test 89.38%**
- 总训练时间: 18.9 min

### 关键教训（Run 1-6 为什么失败）
1. **LR 太低是最致命的错误**: baseline 用 0.1，我们最高只用 1e-2（低 10 倍），backbone lr 只有 5e-4（低 200 倍）。Fine-tune 需要足够大的 lr 才能让 backbone 适应新的细粒度分类任务
2. **两阶段冻结策略是过度设计**: 冻结 backbone 限制了模型适应能力，differential lr 增加了调参复杂度但没有带来收益。直接全参数训练 + 统一 lr，让 SGD 梯度自然决定每层更新幅度
3. **分辨率可以高于预训练，但 lr 必须正确**: Run 6 用 448+lr=5e-4 惨败(41%)，Run 8 用 448+lr=0.05 达 91.78%。高分辨率本身不是问题，lr 太低才是
4. **batch size 和 lr 是绑定的**: batch=128 + lr=0.1 是经过验证的组合，小 batch 需要按比例降 lr
5. **简单方法优先**: 最朴素的配置（SGD + step decay + crop+flip）反而最有效
6. **架构和优化器要配套**: EfficientNet + SGD 只有 91.57%，换 AdamW + cosine 就到 92.99%。ResNet 适合 SGD 高 lr，EfficientNet 适合 AdamW 低 lr

## 模型分析结果（eval.py 输出）

### 最易混淆 Top-5
| True | Predicted | Count |
|------|-----------|-------|
| Audi S5 Coupe 2012 | Audi A5 Coupe 2012 | 14 |
| Audi TTS Coupe 2012 | Audi TT Hatchback 2011 | 13 |
| Audi TT Hatchback 2011 | Audi TTS Coupe 2012 | 12 |
| Dodge Caliber Wagon 2007 | Dodge Caliber Wagon 2012 | 12 |
| Dodge Sprinter Cargo Van 2009 | Mercedes-Benz Sprinter Van 2012 | 12 |

### TTA 结果
| 配置 | 单次推理 | TTA (6 views) |
|------|---------|---------------|
| 224px | 89.38% | 90.32% (+0.95%) |
| 448px | 91.78% | 91.93% (+0.15%) |

- 224px 模型 TTA 收益大（+0.95%），448px 模型 TTA 收益小（+0.15%）
- 高分辨率模型本身已捕捉足够细节，多尺度变换边际收益低

### GradCAM 观察
- 正确预测: 模型关注车身轮廓、进气格栅、尾灯等品牌特征区域
- 错误预测（worst classes）: 注意力分散或聚焦在无关区域（背景、水印）

## 参考
- 论文基准: ResNet-50 + Stanford Cars ≈ 85-90% test accuracy
- Baseline 项目: `PyTorch-Stanford-Cars-Baselines`（ResNet-50 pretrained = 90.0%）
- 我们的最佳结果: **92.99%**（EfficientNet-B4 + AdamW + Cosine）
- 用户有 ResNet-18 CIFAR-10 从零实现经验
