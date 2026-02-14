# 项目总结：迁移学习 Stanford Cars

## 最终结果

**Test Accuracy: 92.99%**（EfficientNet-B4 + AdamW + Cosine Annealing）

论文基准 85-90%，baseline 项目 90.0%，我们超了 3 个点。

训练时间: 56.8 分钟 / 90 epochs（RTX 5080 16GB）

## 实验历程

经历了 10 轮实验，从 41% 到 93%。

| 阶段 | Test Acc | 关键变化 |
|------|----------|----------|
| Run 1-4（AdamW 两阶段） | 49-52% | wd 太重、过度正则化 |
| Run 5-6（SGD 两阶段） | 41-51% | lr 太低 |
| Run 7（参考 baseline） | 89.38% | 单阶段 SGD lr=0.1，突破性进展 |
| Run 8（448px 分辨率） | 91.78% | 高分辨率 + linear scaling |
| Run 9（EfficientNet-B4 + SGD） | 91.57% | 换架构但用错优化器 |
| **Run 10（EfficientNet-B4 + AdamW）** | **92.99%** | **架构+优化器配套** |

## 学到了什么

### 1. 学习率是 fine-tune 最重要的超参数

前 6 轮最高只用到 lr=1e-2，而正确答案是 **lr=0.1**。差了 10-200 倍。

Fine-tune 不是"小心翼翼地微调"——预训练模型需要**大幅度调整**才能从 ImageNet 的粗粒度分类（"轿车 vs 吉普"）适应到 Stanford Cars 的细粒度分类（"BMW M3 Coupe 2012 vs BMW M3 Sedan 2012"）。

### 2. 冻结 backbone 是过度设计

直觉上觉得"先训 fc 再解冻 backbone"更稳妥，实际上：
- 冻结 backbone 限制了特征适应能力
- Differential lr 增加了调参维度但没有收益
- **SGD 的梯度本身就会决定每层该更新多少** —— 底层梯度自然小，高层梯度自然大，不需要人为干预

单阶段、统一 lr、全参数训练是最简单也最有效的方案。

### 3. 高分辨率有效，但前提是 lr 正确

Run 6 用 448+lr=5e-4 只有 41%，Run 8 用 448+lr=0.05 达 91.78%。之前以为"分辨率必须匹配预训练"，实际上高分辨率能帮模型看到更多细节（格栅纹理、车标），关键是 lr 要够大让模型适应新的特征尺度。

### 4. Batch size 和 lr 是绑定的

Linear scaling rule: batch size 减半 → lr 也减半。

batch=128 + lr=0.1 → batch=64 + lr=0.05。

### 5. 架构和优化器要配套

这是 Run 9 vs Run 10 的核心教训：

- **ResNet + SGD** (高 lr=0.1, momentum=0.9, step decay) → 91.78%
- **EfficientNet + SGD** (lr=0.05, step decay) → 91.57%（不匹配）
- **EfficientNet + AdamW** (低 lr=1e-3, wd=1e-2, cosine) → **92.99%**

EfficientNet 用的是 depthwise separable convolution，梯度特性和 ResNet 的标准 Conv 不同。SGD 高 lr 对 ResNet 有效，但对 EfficientNet 优化效率不够。AdamW 的自适应学习率更适合 EfficientNet 的架构特点。

### 6. Step decay vs Cosine Annealing

- **Step decay** 的特点：lr 突然下降，test acc 立刻跳升（epoch 31: 83%→87%），直观易懂
- **Cosine Annealing** 的特点：lr 平滑下降，test acc 持续爬升，后半段（epoch 46-71: 91.8%→93.0%）仍在提升

Cosine 更适合 AdamW（本身就是平滑优化器），Step decay 更适合 SGD（需要明确的阶段切换）。

### 7. 先跑通 baseline，再做改进

前 6 轮的错误在于：还没跑通基本配置，就在尝试各种"高级技巧"。

正确的流程：
1. **先用最朴素的配置复现 baseline**
2. 确认 baseline 能达到预期性能
3. 再逐个实验改进点（分辨率、架构、优化器）

## 最终配置

```
模型: EfficientNet-B4 (ImageNet 预训练)
分辨率: 380
优化器: AdamW (lr=1e-3, weight_decay=1e-2)
调度器: Cosine Annealing
增强: RandomResizedCrop(380) + RandomHorizontalFlip
批大小: 64
训练轮数: 90
加速: AMP + torch.compile + channels_last
```
