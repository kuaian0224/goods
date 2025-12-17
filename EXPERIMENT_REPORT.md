# 实验报告：商品多模态分类（纯 CLIP 最优方案）

## 1. 项目概述
本项目目标是对商品进行多模态分类，输入为图片与文本（标题、描述等），输出为 21 类分类结果。模型训练与推理均基于 PyTorch，支持单折与分层 K 折训练、TTA 与多折概率平均推理。最终最优方案为 **纯 CLIP 分支（OpenCLIP ViT-L/14）**，线上 Kaggle 提交准确率 **0.877**（你提供的最终成绩）。

## 2. 数据与预处理流程
### 2.1 数据字段解析
代码中自动推断列名，位于 `src/data/csv_schema.py`：
- `id_col`：默认 `id`
- `label_col`：优先 `categories`，否则 `label`
- `text_cols`：优先 `title` + `description`，否则 `text`

### 2.2 文本拼接
文本通过 `build_text` 将多个字段拼接为一个序列：
- 逐字段去空、去 NaN
- 按字段顺序拼接，空格连接
- 若无文本字段，则返回空串

### 2.3 图片路径与多图策略
图片路径匹配策略（`ProductDataset.resolve_image_paths`）：
- `<id>.jpg/.png/.jpeg/.webp`
- `<id>_*.{任意后缀}`
多图策略由 `data.multi_image_mode` 控制，默认 `first`：
- `first`：使用第一张
- `random`：训练时随机采样一张
- `mean_pool`：多图均值池化

若图片缺失，则以 **黑色占位图** 代替，确保训练流程不中断。

### 2.4 CLIP 图像预处理与增强
CLIP 分支采用 OpenCLIP 推荐的归一化参数：
- `mean = [0.48145466, 0.4578275, 0.40821073]`
- `std  = [0.26862954, 0.26130258, 0.27577711]`

训练增强（`build_clip_train_tfms`）：
- Resize 到 `img_size + 32`
- RandomResizedCrop 到 `img_size`
- RandomHorizontalFlip
- ToTensor + Normalize

验证/推理增强（`build_clip_val_tfms`）：
- Resize 到 `img_size`
- CenterCrop
- ToTensor + Normalize

推理阶段 TTA：水平翻转后与原图结果平均（`tools/infer.py`）。

### 2.5 CLIP 文本 Tokenizer
CLIP 采用 OpenCLIP tokenizer：
- `max_len = 77`
- 输入为拼接后的文本
- 生成 tokens 作为 `input_ids`
- attention_mask 由 `tokens != 0` 自动生成（用于兼容 batch 格式）

## 3. 模型结构（纯 CLIP）
### 3.1 Backbone
使用 OpenCLIP 的 **ViT-L/14** 作为图文编码器（`clip_pretrained: openai`）：
- 图像编码：`clip_model.encode_image`
- 文本编码：`clip_model.encode_text`

### 3.2 Fusion 方式
图像与文本编码后的向量 **直接拼接（concat）**：
```
fused = concat(img_feat, txt_feat)
```
无 cross-attention，无额外投影层。

### 3.3 分类头
使用 MLP 分类头（`hidden_dim = 512`）：
- Dropout
- Linear -> GELU -> Dropout -> Linear

## 4. 训练策略与实现细节
### 4.1 K 折策略
默认 5 折分层 K-Fold（`folds: 5`），若数据中已有 `fold` 列则直接使用。每折保存 `best.pt`，最终支持折内平均推理。

### 4.2 损失函数与指标
- 损失：CrossEntropyLoss
- label_smoothing：CLIP 分支为 `0.0`
- 监控指标：val top1 / top3 accuracy（写入 TensorBoard 与 CSV）

### 4.3 优化器与学习率分组
优化器：AdamW  
支持按模块分组学习率（`train.lrs`），CLIP 分支在 Stage2 细化为：
- visual（image_encoder）：1e-5
- text（text_encoder）：1e-5
- head：3e-4

### 4.4 Scheduler
使用 `transformers.get_linear_schedule_with_warmup`，默认 `warmup_ratio = 0.0`（可在配置中开启）。

### 4.5 AMP 与梯度累积
`train.amp: true`，混合精度训练；默认 `grad_accum = 1`。

### 4.6 早停
若验证集 top1 accuracy 连续 `early_stop_patience` 轮未提升则提前停止（默认 2）。

## 5. 两阶段微调方案（最终最佳）
### 5.1 阶段 1：冻结 CLIP，只训练 head
配置文件：`configs/clip_fusion_stage1_best_stage_v1.yaml`
- `clip_trainable: false`
- `hidden_dim: 512`
- `img_size: 224`
- `max_len: 77`
- `epochs: 5`
- `batch_size: 64`
- `lr/head: 1e-3`

目的：让分类头先稳定收敛，避免一开始扰动 CLIP 表征。

### 5.2 阶段 2：解冻 CLIP，小学习率微调
配置文件：`configs/clip_fusion_stage2 _best_stage_v2.yaml`
- `clip_trainable: true`
- `hidden_dim: 512`
- `img_size: 224`
- `epochs: 20`
- `batch_size: 32`
- `init_ckpt: checkpoints/clip_fusion_stage1_v1/fold{fold}/best.pt`
- `lrs`：
  - image_encoder：1e-5
  - text_encoder：1e-5
  - head：3e-4

目的：在不破坏 CLIP 泛化能力的前提下，微调至任务分布。

## 6. 推理与提交
### 6.1 纯 CLIP 推理
```
python tools/infer.py --config "configs/clip_fusion_stage2 _best_stage_v2.yaml"
```
- 自动读取 `ckpt_dir` 下 `fold*/best.pt`
- 多折概率平均
- TTA（水平翻转）默认开启

### 6.2 多尺度（可选）
复制配置，改 `data.img_size: 256` 或 `240`，分别推理后用 `tools/infer_ensemble.py` 做平均，提升鲁棒性。

## 7. 中间结果与最终结果
训练过程的中间指标记录位置：
- `logs/{exp_name}/foldX.csv`：每 epoch 的 `train_loss/val_acc/val_top3`
- `logs/{exp_name}/foldX_tb/`：TensorBoard 曲线

最终最佳方案为 **纯 CLIP Stage2**，线上 Kaggle 提交准确率 **0.877**。

## 8. 复现步骤（单折示例）
```
pip install -r requirements.txt

# 阶段 1
python tools/train.py --config configs/clip_fusion_stage1_best_stage_v1.yaml --fold 0

# 阶段 2（自动加载阶段1 best）
python tools/train.py --config "configs/clip_fusion_stage2 _best_stage_v2.yaml" --fold 0

# 推理
python tools/infer.py --config "configs/clip_fusion_stage2 _best_stage_v2.yaml"
```

## 9. 方案总结
本项目最终采用 **OpenCLIP ViT-L/14 纯 CLIP 方案**，核心优势在于：
- 预训练视觉-文本对齐带来强泛化能力；
- 两阶段训练降低微调风险；
- 小学习率微调 CLIP 主体，保持稳定收敛；
- 结合 TTA 和多折平均提升线上成绩。
