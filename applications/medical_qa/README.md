# 方案5：医疗问答域应用 Demo

## 项目目标

在 MiniMind（26M 参数微型 LLM）基础上，通过 LoRA 微调构建一个面向医疗领域的问答助手。本方案完整演示了从数据准备、领域微调、效果评测到交互式 Demo 的全流程，验证小模型在垂直领域的可落地性。

**核心思路：**
- 以 MiniMind 的 `full_sft` 权重为基础模型
- 使用医疗问答数据集进行 LoRA 参数高效微调
- 通过多维度评测（BLEU、ROUGE-L、distinct-n、安全性）量化微调效果
- 提供 Streamlit 交互式 Demo 供体验

## 目录结构

```
applications/medical_qa/
├── README.md                    # 本文档
├── run_experiment.py            # 一键实验脚本（主入口）
│
├── data/                        # 数据相关
│   ├── prepare_data.py          # 数据下载与预处理脚本
│   ├── medical_sft.jsonl        # LoRA 微调训练数据（自动生成）
│   ├── eval.jsonl               # 评测数据集 conversations 格式（自动生成）
│   └── eval_qa.json             # 评测数据集 question/answer 格式（自动生成）
│
├── eval/                        # 评测相关
│   └── medical_eval.py          # 评测脚本：Base vs LoRA 对比
│
├── demo/                        # Demo 相关
│   └── medical_chat.py          # Streamlit 医疗问答 Demo
│
└── results/                     # 结果输出
    ├── eval_results.json        # 评测指标 JSON 文件
    └── report.md                # 实验报告（自动生成）
```

## 快速开始

### 环境准备

确保已安装项目依赖：

```bash
cd D:\Year3\科研\minimind-master
pip install -r requirements.txt
pip install streamlit   # 如果需要运行 Demo
```

确保已具备基础模型权重：

```bash
# 检查是否存在 full_sft 权重
ls out/full_sft_512.pth
# 如果不存在，需要先完成预训练和 SFT 流程
```

### 1. 数据准备

数据准备脚本会自动下载/生成医疗问答数据并拆分为训练集和评测集：

```bash
cd applications/medical_qa
python data/prepare_data.py
```

生成的数据格式为 JSONL，每行一个 JSON 对象：

```json
{"conversations": [{"role": "user", "content": "高血压患者应该注意什么？"}, {"role": "assistant", "content": "高血压患者应注意以下几点..."}]}
```

### 2. LoRA 微调（可选，需要 GPU）

**方式一：使用一键脚本**

```bash
cd applications/medical_qa
python run_experiment.py --epochs 5 --batch_size 16 --learning_rate 1e-4
```

**方式二：手动调用训练脚本**

```bash
cd D:\Year3\科研\minimind-master
python -m trainer.train_lora \
    --lora_name lora_medical \
    --data_path applications/medical_qa/data/medical_sft.jsonl \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --save_dir out/lora \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --from_weight full_sft
```

训练完成后，LoRA 权重保存在 `out/lora/lora_medical_512.pth`。

### 3. 运行评测

评测脚本会自动对比 Base 模型和 LoRA 微调模型在医疗问答上的表现：

```bash
cd applications/medical_qa
python run_experiment.py --skip_train    # 跳过训练，仅评测+报告
```

或手动运行：

```bash
python eval/medical_eval.py \
    --base_weight full_sft \
    --lora_name lora_medical \
    --hidden_size 512 \
    --eval_data data/medical_eval.jsonl \
    --output_dir results/
```

评测结果会保存到 `results/eval_results.json`，并自动生成 Markdown 报告 `results/report.md`。

### 4. 启动 Demo

```bash
cd applications/medical_qa
python run_experiment.py --demo
```

或直接启动 Streamlit：

```bash
streamlit run demo/medical_chat.py
```

浏览器将自动打开 `http://localhost:8501`，可与医疗问答助手交互。

### 一键运行全部步骤

```bash
python run_experiment.py
```

这将依次执行：数据准备 -> LoRA 微调 -> 评测对比 -> 生成报告。可通过参数跳过任意步骤：

```bash
python run_experiment.py --skip_train              # 跳过训练
python run_experiment.py --skip_eval               # 跳过评测
python run_experiment.py --skip_train --skip_eval  # 仅生成报告
```

## 评测指标说明

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **BLEU-4** | 生成文本与参考答案的 4-gram 重叠度 | 基于 n-gram 精度的几何平均，值越高表示生成内容越接近参考 |
| **ROUGE-L** | 最长公共子序列比率 | 衡量生成文本与参考答案的序列匹配程度，关注召回率 |
| **distinct-1** | 一元组多样性 | 唯一 unigram 数量 / 总 unigram 数量，反映用词丰富度 |
| **distinct-2** | 二元组多样性 | 唯一 bigram 数量 / 总 bigram 数量，反映表达多样性 |
| **安全性检测** | 医疗安全关键词覆盖 | 检查回答是否包含必要的安全提示（如"请咨询医生"等） |

**指标解读：**
- BLEU-4 和 ROUGE-L 越高，说明生成的回答与标准答案越接近
- distinct-1/2 越高，说明回答用词越丰富、不重复
- 安全性检测是医疗场景的特殊要求，确保模型不会给出危险建议

## 实验结果

详细评测报告见：

- [evidence/report_official.md](../../evidence/report_official.md)
- [evidence/report_sample100.md](../../evidence/report_sample100.md)

| 实验 | 训练数据 | BLEU | ROUGE-L | 评测样本 |
|---|---|---:|---:|---:|
| Base (`full_sft_512`) | — | 0.0248 | 0.1243 | 20 |
| Medical LoRA (official) | MiniMind `lora_medical.jsonl` | 0.0243 | **0.1283** | 20 |
| Base (`full_sft_512`) | — | 0.0267 | 0.1239 | 20 |
| Medical LoRA (sample-100) | 80 条合成问题 | **0.0288** | **0.1460** | 20 |

> 评测集只有 20 条，指标波动大，仅作为流程验证。

## 可解释性分析

医疗问答模型的可解释性分析见：

- [../interpretability/README.md](../interpretability/README.md)

该模块在冻结 MiniMind 隐状态上训练科室分类探针，并使用 token occlusion、LIME、KernelSHAP 做词级归因；在项目根目录运行 `python applications/interpretability/run_all.py` 可复现，结果报告在 `applications/interpretability/results/report.md`。

## 已知限制

### 当前方案的不足

1. **模型容量有限：** MiniMind 仅 26M 参数，对复杂医疗推理（如多症状联合诊断）能力不足
2. **数据规模受限：** 微调数据量有限，覆盖面可能不够广，对罕见疾病支持较弱
3. **缺乏知识检索：** 纯生成式方案没有接入外部医疗知识库，可能产生不准确的回答
4. **安全性保障不完善：** 安全性检测基于关键词匹配，无法覆盖所有风险场景

### 改进方向

1. **RAG 增强：** 接入医疗知识库（如用药指南、诊疗规范），通过检索增强生成提升准确性
2. **更大基座模型：** 将方案迁移到更大参数量的模型上，提升复杂推理能力
3. **多轮对话优化：** 引导模型主动询问症状细节，模拟医生问诊流程
4. **强化学习对齐：** 使用 DPO/GRPO 等方法，基于医生反馈优化回答质量
5. **安全护栏：** 添加规则引擎或安全分类器，过滤高风险回答
6. **评测完善：** 引入医生人工评测，补充自动指标无法覆盖的维度（如专业性、同理心）
