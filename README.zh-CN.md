[![English](https://img.shields.io/badge/README-English-2f81f7?style=flat-square)](README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d73a4a?style=flat-square)](README.zh-CN.md)

# MiniMind 医疗问答 + 可解释性（26M）

**这是一个最小可运行版本**，基于 **26M 参数的 MiniMind**，聚焦中文医疗问答和可解释性分析。

本仓库基于开源项目 [MiniMind](https://github.com/jingyaogong/minimind)（Apache-2.0）。仓库包含验证模型、训练管线、医疗 LoRA 实验和可解释性分析所需的代码。**大型基座权重和数据集不在仓库内**，下载方式见下文。

> **最小可运行版本。** 本仓库是完整项目的一个子集：包含代码、医疗 LoRA 适配器和可解释性结果，足以端到端验证整条流程。仓库**不包含** 56MB 基座 checkpoint、大型训练数据集，也不包含完整的 pretrain/SFT/DPO checkpoint 和训练日志。具体边界见[范围与限制](#范围与限制)。

## 这个最小版本包含什么

- **26M decoder-only Transformer**：RoPE、RMSNorm、SwiGLU、GQA、causal self-attention、KV-Cache、Flash Attention。
- **训练管线代码**：tokenization、next-token 预训练、SFT、DPO、LoRA。
- **医疗 LoRA 适配器**（`0.131M` 可训练参数，约 `0.50%`）和评测报告。
- **可解释性工具**：冻结 MiniMind hidden state → 8 科室线性探针 → 5-fold 交叉验证 + one-vs-rest AUC → token occlusion + LIME + KernelSHAP，并解释 out-of-fold 预测。
- **合成医疗示例数据**和可复现报告。

### 这个最小版本不包含什么

- 56MB 基座 checkpoint `full_sft_512.pth` —— 使用“快速开始”里的命令下载。
- 大型数据集（`pretrain_hq.jsonl`、`sft_mini_512.jsonl`、`dpo.jsonl`、`lora_medical.jsonl`）—— 从 MiniMind 数据集链接下载。
- 完整的 pretrain/SFT/DPO checkpoint 和训练日志；本仓库的医疗结果使用官方 SFT 基座 + 仓库内的医疗 LoRA 适配器。
- 64M checkpoint、3B-token 预训练 checkpoint、10K+ 指令 SFT checkpoint —— 代码支持这些训练，但它们不属于这个最小版本。

## 结果

### 模型

- 默认配置（`hidden_size=512`、`num_hidden_layers=8`、`vocab_size=6400`）下共 **25.83M 参数**。

### 医疗问答 LoRA

| 实验 | 训练数据 | BLEU | ROUGE-L | 评测样本 |
|---|---|---:|---:|---:|
| Base（`full_sft_512.pth`） | — | 0.0248 | 0.1243 | 20 |
| Medical LoRA（official） | MiniMind `lora_medical.jsonl` | 0.0243 | **0.1283** | 20 |
| Base（`full_sft_512.pth`） | — | 0.0267 | 0.1239 | 20 |
| Medical LoRA（sample-100） | 80 条合成问题 | **0.0288** | **0.1460** | 20 |

评测集只有 20 条，指标波动较大，仅作为流程验证，不应视为 benchmark。完整报告在 [`evidence/`](evidence/)。

### 可解释性

- 66 条带科室标签的合成医疗问题，覆盖 8 个科室。
- **8 分类 out-of-fold 准确率：0.636**（随机猜测 0.125）；Macro-F1：**0.642**。
- 单类 one-vs-rest AUC：**0.729 – 0.961**（因科室而异）。
- 三种方法（occlusion / LIME / KernelSHAP）的 top-1 线索一致：
  - 儿科：`宝宝`（occlusion +0.952、LIME +0.884、SHAP +0.922）。
  - 妇产科：`宫颈`（occlusion +0.887、LIME +0.343、SHAP +0.298）。
- 次要词的归因可能不一致；报告里如实记录，没有隐藏分歧。

完整报告：[`applications/interpretability/results/report.md`](applications/interpretability/results/report.md)。

## 快速开始

这是最小版本的完整验证路径：安装依赖 → 下载基座 checkpoint → 验证模型/LoRA → 运行可解释性分析。

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
python -m pip install -r applications/interpretability/requirements.txt
```

### 2. 下载基座 checkpoint（不在仓库内，约 56MB）

```bash
mkdir -p out
curl -L -o out/full_sft_512.pth \
  https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch/resolve/master/full_sft_512.pth
```

### 3. 验证

```bash
python scripts/verify_setup.py --weight_path out/full_sft_512.pth
```

预期输出包含：

```text
[ok] tokenizer loaded: vocab_size=6400
[ok] model instantiated: 25.83M parameters
[ok] forward pass: logits=(1, 8, 6400)
[ok] loaded base checkpoint: full_sft_512.pth
[ok] loaded LoRA adapter: lora_medical_official_512.pth
[ok] LoRA parameters: 0.131M
[done] minimal setup is runnable
```

### 4. 运行可解释性分析

```bash
python applications/interpretability/run_all.py
```

结果（图表、JSON 指标和 `report.md`）写入 `applications/interpretability/results/`。笔记本 CPU 上约 15–20 秒。

## 仓库结构

```text
model/                          # MiniMind 架构 + LoRA
dataset/                        # Pretrain / SFT / DPO 数据集类
trainer/                        # Pretrain、SFT、DPO、LoRA 训练脚本
scripts/
  ├── train_tokenizer.py        # BPE tokenizer 训练（vocab 6400）
  └── verify_setup.py           # 最小可运行验证
applications/
  ├── medical_qa/               # 数据准备、LoRA 实验、评测、Streamlit Demo
  └── interpretability/         # probe + occlusion/LIME/KernelSHAP + 生成报告
evidence/                       # 仓库内包含的 LoRA 适配器 + 评测 JSON/Markdown
```

## 训练管线（可选）

训练脚本已包含在仓库中。从
[HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset) 或
[ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset) 下载数据集，放到 `dataset/` 下：

| 文件 | 大小 | 用途 |
|---|---:|---|
| `pretrain_hq.jsonl` | ~1.6GB | next-token 预训练 |
| `sft_mini_512.jsonl` | ~1.2GB | SFT 指令数据 |
| `dpo.jsonl` | ~55MB | DPO 偏好对 |
| `lora_medical.jsonl` | ~34MB | 医疗 LoRA |

然后在 `trainer/` 目录下运行：

```bash
cd trainer

# Next-token 预训练（多 epoch；代码支持 3B-token 规模，但本仓库不含该 checkpoint）
python train_pretrain.py \
  --data_path ../dataset/pretrain_hq.jsonl \
  --epochs 10 --hidden_size 512 --num_hidden_layers 8 --save_dir ../out

# 监督微调
python train_full_sft.py \
  --data_path ../dataset/sft_mini_512.jsonl \
  --epochs 2 --hidden_size 512 --num_hidden_layers 8 \
  --from_weight pretrain --save_dir ../out

# DPO
python train_dpo.py \
  --data_path ../dataset/dpo.jsonl \
  --epochs 1 --hidden_size 512 --num_hidden_layers 8 \
  --from_weight full_sft --save_dir ../out

# 医疗 LoRA
python train_lora.py \
  --lora_name lora_medical \
  --data_path ../dataset/lora_medical.jsonl \
  --epochs 3 --hidden_size 512 --num_hidden_layers 8 \
  --from_weight full_sft --save_dir ../out
```

> 仓库包含训练管线代码；本仓库发布的实验结果使用官方 `full_sft_512.pth`
> 基座 + 仓库内的医疗 LoRA 适配器，以及可解释性探针。完整的 3B-token 预训练
> 代码支持运行，但其 checkpoint 不在本仓库内。

## 数据来源

- `applications/medical_qa/data/sample_medical_qa.jsonl` 是**合成数据**：由
  `generate_sample_data.py` 生成 100 条 QA（其中 66 条带 8 科室标签）。仅用于开发和流程验证，不是真实临床数据。
- 医疗 LoRA 训练使用官方 MiniMind `lora_medical.jsonl`（约 34MB），**不随仓库分发**。
  请从上面的 MiniMind 数据集链接下载。
- 仓库不包含任何可识别患者身份的数据。

## 范围与限制

- 本仓库是 **26M** 模型，不包含 64M checkpoint。如果要用同一套代码做 ~64M 变体，
  `--hidden_size 832 --num_hidden_layers 8` 是 **63.91M** 参数，前向可以跑通，但必须从头预训练。
- 不包含 3B-token 预训练 checkpoint 或训练日志。训练脚本支持多 epoch 预训练，
  但该运行不属于本次发布内容。
- 不包含完整 SFT / DPO checkpoint；基座 `full_sft_512.pth` 从官方 MiniMind release 下载。
- 可解释性分析解释的是 **冻结 MiniMind 表征上的线性探针**，不是生成式语言头本身。
- 医疗评测集只有 20 条，指标仅作为 smoke test。
- 探针数据是合成的，结果用于验证可解释性流程，不构成临床结论。

## 署名与许可证

本仓库基于 [MiniMind](https://github.com/jingyaogong/minimind)（作者 Jingyao Gong），
使用 Apache License 2.0。原始 [`LICENSE`](LICENSE) 已包含在仓库中。
医疗问答应用、可解释性模块、仓库内的 LoRA 适配器和生成结果是在本项目中的新增内容，
同样使用 Apache-2.0 许可证。
