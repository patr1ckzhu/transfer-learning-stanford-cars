# 项目总结：ResNet-50 迁移学习 Stanford Cars

## 最终结果

**Test Accuracy: 89.38%**（论文基准 85-90%，baseline 项目 90.0%）

训练时间: 18.9 分钟 / 90 epochs（RTX 5080 16GB）

## 走过的弯路

经历了 7 轮实验，前 6 轮全部失败（41-52%），第 7 轮参考开源 baseline 后一次成功。

| 阶段 | Test Acc | 问题 |
|------|----------|------|
| Run 1-4（AdamW 各种调参） | 49-52% | weight decay 太重、过度正则化 |
| Run 5-6（SGD 两阶段） | 41-51% | lr 太低、448 分辨率有害 |
| **Run 7（参考 baseline）** | **89.38%** | 单阶段 SGD lr=0.1，一次到位 |

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

### 3. 输入分辨率必须匹配预训练

ResNet-50 预训练于 224x224。尝试 448 后效果反而更差：
- Phase 1 frozen backbone: 224→41% vs 448→19%
- 最终: 224→52% vs 448→41%

预训练的卷积核、BN 统计量都是基于 224 学的，强行改分辨率会破坏这些特征。

### 4. Batch size 和 lr 是绑定的

Linear scaling rule: batch size 翻倍 → lr 也翻倍。

batch=128 + lr=0.1 是验证过的组合。batch=16 + lr=5e-4 的学习信号太弱，模型学不动。

### 5. Step decay 简单有效

不需要 cosine annealing、warmup 等花哨的 scheduler。

Step decay（每 30 epochs ÷10）的效果非常直观：
- Epoch 1-30 (lr=0.1): 大步探索，test 从 0% → 83%
- Epoch 31 (lr=0.01): **lr 一降，test 立刻从 83% 跳到 87%** —— 模型已经在好的区域了，降 lr 就能收敛
- Epoch 61-90 (lr=0.001): 精细微调，最终 89.4%

### 6. 数据增强：少即是多

只用了最基础的 RandomResizedCrop + RandomHorizontalFlip，去掉了 ColorJitter、Dropout、Label Smoothing。

对于 8144 张图 + 预训练模型的组合，过多增强和正则化反而导致 underfitting。

### 7. 先跑通 baseline，再做改进

前 6 轮的错误在于：还没跑通基本配置，就在尝试各种"高级技巧"（两阶段训练、differential lr、高分辨率、重度增强）。

正确的流程：
1. **先用最朴素的配置复现 baseline**
2. 确认 baseline 能达到预期性能
3. 再逐个实验改进点

## 最终配置

```
模型: ResNet-50 (ImageNet V2 预训练)
分辨率: 224
优化器: SGD (lr=0.1, momentum=0.9, weight_decay=1e-4)
调度器: Step decay ÷10 every 30 epochs
增强: RandomResizedCrop(224) + RandomHorizontalFlip
批大小: 128
训练轮数: 90
加速: AMP + torch.compile + channels_last
```
