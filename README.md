# 商品多模态分类项目（图片+文本）

基于 PyTorch 的多模态分类工程，支持图片-only、文本-only、图片+文本融合，单折/分层 K 折训练，TTA 与多折概率平均推理，生成 Kaggle 形式的 `submission.csv`。

## 目录结构
```
goods/
  configs/            # 配置
  data/               # 数据（需自行放置）
    train.csv
    test.csv
    sample_submission_products.csv
    train_images/
    test_images/
  src/                # 核心代码
  tools/              # CLI 脚本
  checkpoints/        # 训练输出（运行后生成）
  logs/               # 训练日志（运行后生成）
  outputs/            # sanity 示例（运行后生成）
```

## 环境准备
- Python 3.10+
- 进入 `goods/` 安装依赖：`pip install -r requirements.txt`

## 数据要求与字段适配
- 必须列：`id`
- 标签优先：`categories`，其次 `label`
- 文本优先：`title` + `description`，否则尝试 `text`，缺失则为空串
- 图片匹配：按 `<id>.jpg/png/jpeg/webp` 或 `<id>_*.jpg` 搜索
- 多图策略：`data.multi_image_mode` 支持 `first`、`random`、`mean_pool`

## 核心配置说明（以 configs/fusion.yaml 为例）
- `exp_name`：实验名（用于日志/ckpt 子目录）
- `fast_sanity`：true 时 `sanity_check` 跳过全量缺失统计
- `data.*`：数据根目录、csv 名、图片目录、文本列、图像大小、工作线程数等
- `model.type`：`image` / `text` / `fusion`
- `model.image_backbone`：如 `convnext_tiny`
- `model.text_backbone`：如 `bert-base-multilingual-cased`
- `model.fusion`：`concat` / `sum` / `gated_concat`
- `train.*`：折数、指定折、epoch、batch、`lr`（基准 lr）、`lrs`（可选字典，支持 `image_encoder` / `text_encoder` / `head` / `others` / `default` 分别设置 lr）、weight_decay、AMP、梯度累积、label smoothing、早停等；可选 `ckpt_dir` 自定义保存路径
- `infer.*`：推理 batch、TTA、`ckpt_dir`（扫描 `fold*/best.pt`）、输出文件名

示例（融合模型分模块 lr）：
```
train:
  lr: 2e-5
  lrs:
    image_encoder: 1e-5
    text_encoder: 2e-5
    head: 3e-5
```

## 快速检查（可选）
```
python tools/sanity_check.py --config configs/fusion.yaml
```
- 抽样 16 条打印文本/标签/图片存在性，保存示例图到 `outputs/sanity/`
- 若设置 `fast_sanity: true`，跳过全量缺失统计，加快运行

## 训练流程
### 单折调试
```
python tools/train.py --config configs/fusion.yaml --fold 0
```
### 全部 K 折训练
```
python tools/train.py --config configs/fusion.yaml --fold -1
```
- 自动 Stratified K-Fold；若 csv 已有 `fold` 列则直接使用
- 模型/日志保存到 `train.ckpt_dir`（若未设置则用 `infer.ckpt_dir`，否则 `checkpoints/{exp_name}`）和 `logs/{exp_name}`

### 其他模型类型示例
- 图片-only：`python tools/train.py --config configs/baseline_image.yaml --fold 0`
- 文本-only：`python tools/train.py --config configs/baseline_text.yaml --fold 0`

### CLIP 两阶段拉分方案（ViT-L/14）
- 阶段 1：冻结 CLIP，只训练分类头（更稳）。`python tools/train.py --config configs/clip_fusion_stage1.yaml --fold 0`（或 `-1` 全折）。hidden_dim=512，lr/head=1e-3，epochs=5。
- 阶段 2：解冻 CLIP，小 lr 微调 + head 中等 lr。`python tools/train.py --config configs/clip_fusion_stage2.yaml --fold 0`。配置里 `train.init_ckpt` 会自动从阶段 1 对应折的 best.pt 预热；image/text lr=1e-5，head lr=3e-4，batch=48。
- `configs/clip_fusion.yaml` 为阶段 1 的简化版，便于快速跑单阶段。
- 显存吃紧可调低 batch 或继续冻结（`clip_trainable: false`）；若端到端微调，视显存降低 `lrs.head`、`train.batch_size`。

## 推理与提交
```
python tools/infer.py --config configs/fusion.yaml
```
- 自动加载 `cfg.infer.ckpt_dir` 下的 `fold*/best.pt`（或单个 best.pt）
- 多折概率平均，TTA 可选，输出 `submission.csv`

### 多模型/多尺度融合
- 使用 `tools/infer_ensemble.py` 对多个配置的概率做加权平均：
  ```
  python tools/infer_ensemble.py --configs configs/fusion.yaml configs/clip_fusion_stage2.yaml --weights 1 1 --output submission_ensemble.csv
  ```
- CLIP 建议多尺度：复制 `configs/clip_fusion_stage2.yaml` 改 `data.img_size: 256`（或 240），再跑一遍 `tools/infer.py`，与 224 尺度或 convnext+deberta 分支一起融合。

## 提速建议（大显卡如 5090）
- `train.batch_size` 增大到 64/96/128（OOM 再降或用 `grad_accum`）
- `data.num_workers` 提升到 8/12，开启 `pin_memory/persistent_workers`（代码已默认 pin）
- 降低 `data.img_size`（如 160/192）或用更轻 backbone
- 减少 `train.epochs`、先单折跑通再全折

## 常见问题
- **图片缺失**：检查图片是否放在 `data/train_images`、`data/test_images`，后缀是否匹配；缺失会用占位图并警告
- **字段不匹配**：在配置里显式设置 `data.id_col` / `label_col` / `text_cols`
- **ckpt 位置**：优先 `train.ckpt_dir`，否则 `infer.ckpt_dir`，否则 `checkpoints/{exp_name}`
- **AMP 报警**：已切换 `torch.amp` 接口，确保 PyTorch 版本 >= 2.x
