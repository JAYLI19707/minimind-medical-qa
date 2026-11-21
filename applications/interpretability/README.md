# MiniMind 医疗问答可解释性分析

在 MiniMind2-Small（26M）的冻结隐状态上训练一个科室分类探针，并用 **token occlusion、LIME、KernelSHAP** 解释探针的预测，提供一条可复现的特征重要性 / 可解释性分析流程。

## 1. 这个模块做什么

给定一条医疗问题，预测它属于 8 个科室中的哪一个：

`内科 / 外科 / 儿科 / 妇产科 / 皮肤科 / 眼科 / 口腔科 / 中医科`

然后回答三个问题：

1. **全局**：哪些词总体上支持某个科室的判断？
2. **局部**：对某一条具体问题，哪些词把它推向当前预测？哪些词在反对？
3. **方法对比**：删除一个词（occlusion）、局部线性近似（LIME）、Shapley 值（SHAP）三种解释是否一致？

## 2. 为什么用“探针”而不是直接解释生成结果

直接对自回归生成做 SHAP/LIME 需要先定义一个稳定的标量目标（例如某个答案 token 的概率），而小模型的自由生成方差很大，解释结果不稳定。这里采用 NLP 可解释性中常用的 **probing** 思路：

- 冻结 MiniMind 全部参数；
- 取最后一层 hidden state 的 mean pooling 作为句子表征；
- 只训练一个线性分类器（逻辑回归）。

这样预测目标明确、可复现，SHAP/LIME 的归因也有清晰的语义：**MiniMind 的隐状态里，哪些输入词支持科室判断**。

> 注意：解释对象是“基于 MiniMind 表征的探针”，不是生成式语言头本身。这个边界在报告中也明确说明。

## 3. 方法

| 步骤 | 实现 |
|---|---|
| 表征 | 冻结 MiniMind，最后一层 hidden state 按 attention mask 做 mean pooling（512 维） |
| 探针 | `StandardScaler` + `LogisticRegression(class_weight='balanced')` |
| 评测 | 5-fold `StratifiedKFold`，out-of-fold 准确率 / macro-F1 / 混淆矩阵；每个科室单独训练 one-vs-rest 探针并报告 AUC |
| 全局重要性 | leave-one-word-out occlusion：删除一个词，看目标科室概率下降多少，按科室聚合 |
| 局部解释 | LIME（`LimeTextExplainer`）+ KernelSHAP（词存在性 mask） |
| 中文分词 | jieba，去掉标点和高频停用词 |

被解释的样本全部使用 **out-of-fold** 预测：解释某条样本时，使用的是没有见过这条样本的 fold 模型。

### 实现细节

- **mean pooling**：初测中 mean pooling 的 8 分类 CV 准确率明显高于 last-token pooling（约 0.61 vs 0.25），因此选用 mean pooling。
- **LIME 的空格问题**：LIME 用空格分词，但送入探针前会去掉空格，保持与探针训练时的中文输入一致；否则 LIME 与 SHAP 的归因差异会明显变大。
- **停用词**：LIME/SHAP 的特征空间保留停用词，保证有足够的扰动特征；展示时过滤掉停用词。occlusion 直接使用内容词。
- **out-of-fold**：被解释样本使用未见过该样本的 fold 模型，避免解释 in-sample 预测。

## 4. 快速开始

在项目根目录执行：

```bash
# 1) 安装额外依赖（torch / transformers 复用主环境）
pip install -r applications/interpretability/requirements.txt

# 2) 生成带 department 标签的示例数据
python applications/medical_qa/data/generate_sample_data.py

# 3) 下载官方 26M SFT 权重（约 56MB）
mkdir -p out
curl -L -o out/full_sft_512.pth \
  https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch/resolve/master/full_sft_512.pth

# 4) 运行完整分析（CPU 即可，本机约 20-30 秒）
python applications/interpretability/run_all.py
```

> 本机调试时使用了项目根目录的 `.venv`（`--system-site-packages` 复用 Anaconda 的 torch/transformers）。
> 如果 `python` 指向的环境没有 shap/lime，可以这样运行：
> `.venv/Scripts/python.exe applications/interpretability/run_all.py`

常用参数：

```bash
python applications/interpretability/run_all.py \
  --weight_path out/full_sft_512.pth \
  --n_splits 5 \
  --C 1.0 \
  --num_lime_samples 200 \
  --num_shap_samples 150 \
  --max_examples 3
```

## 5. 输出

所有结果写入 `applications/interpretability/results/`：

| 文件 | 内容 |
|---|---|
| `report.md` | 完整报告：指标、全局关键词、单样本解释、方法对比、局限 |
| `probe_metrics.json` | 8 分类 out-of-fold 准确率、macro-F1、混淆矩阵、分类报告 |
| `one_vs_rest_auc.json` | 每个科室的二分类探针 AUC |
| `global_occlusion.json` | 每个科室的 top supporting tokens |
| `probe_confusion_matrix.png` | 混淆矩阵 |
| `probe_auc.png` | 单类 AUC 条形图 |
| `global_top_tokens.png` | 8 个科室的全局关键词 |
| `occlusion_*.png` | 单样本 token occlusion 归因 |
| `lime_*.png` / `lime_*.html` | LIME 局部解释（PNG + 交互式 HTML） |
| `shap_*.png` | 单样本 SHAP 归因 |
| `shap_summary.png` | 解释样本的全局 SHAP 摘要 |

### 本次运行结果（CPU，约 15-20 秒）

| 指标 | 结果 |
|---|---|
| 数据 | 66 条有科室标签的合成医疗问题，8 个科室 |
| 8 分类 out-of-fold 准确率 | **0.636**（随机猜测 0.125） |
| Macro-F1 | **0.642** |
| 单类 one-vs-rest AUC | **0.729（外科）～ 0.961（妇产科）** |
| 最强单样本解释 | 儿科问题中 `宝宝`：occlusion +0.952、SHAP +0.922、LIME +0.884，三种方法均排第 1 |
| 第二种一致案例 | 妇产科问题中 `宫颈`：occlusion +0.887、SHAP +0.298、LIME +0.343，三种方法均排第 1 |

皮肤科样本里，三种方法都把 `疼 / 走路` 排在前列；但 occlusion 还强调 `去除 / 脚上 / 鸡眼`，SHAP/LIME 则给 `鸡眼` 负权重。妇产科样本也有类似现象：三种方法都把 `宫颈` 排第 1，但 LIME/SHAP 给 `糜烂` 负权重，occlusion 给正贡献。这说明词级归因会受到词间交互和扰动方式影响，不能把单一方法的结果当成因果结论。

> LIME 是局部线性近似，绝对值和 occlusion/SHAP 不完全可比；优先比较排名和符号。

## 6. 结果怎么读

- **8 分类准确率**：在 66 条有标签样本上做 5-fold，随机猜测是 12.5%。数据量小，绝对值不要过度解读。
- **单类 one-vs-rest AUC**：每个科室单独训练二分类探针，衡量“能不能把这个科室和其他科室分开”。这个指标比 8 分类准确率更稳定。
- **Occlusion / LIME / SHAP 一致性**：如果三种方法都把“宝宝 / 孩子”排到儿科问题的前列，说明这个探针确实抓到了可解释的科室线索；如果不一致，说明词间交互或扰动敏感性较强。
- **不要宣称因果**：这些方法是归因/相关性分析，不是因果推断。

## 7. 已知局限

1. 数据是项目内置的合成医疗问答（66 条带科室标签），规模小，指标波动大。
2. 解释对象是线性探针，不是生成式语言头；不能直接等同于“模型生成答案时关注了什么”。
3. 词级扰动会破坏原始语序和语法，LIME/SHAP 是局部近似。
4. 中文分词和停用词表会影响归因粒度，换分词器结果会变化。
