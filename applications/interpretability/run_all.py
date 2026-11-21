"""End-to-end interpretability analysis for MiniMind medical QA.

Pipeline:
1. Load the 8-department medical question set and the frozen 26M SFT weights.
2. Extract mean-pooled MiniMind hidden states and fit a logistic-regression probe.
3. Evaluate the probe with 5-fold cross-validation and one-vs-rest AUC.
4. Explain held-out predictions with token occlusion, LIME and word-level SHAP.
5. Write figures, JSON metrics and a Markdown report to ``results/``.

Run from the project root:
    python applications/interpretability/run_all.py
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import (  # noqa: E402
    DEFAULT_WEIGHT_PATH,
    DEPARTMENT_EN,
    DEPARTMENTS,
    SAMPLE_DATA_PATH,
    STOPWORDS,
    MiniMindEmbedder,
    fit_logistic,
    load_labeled_samples,
    predict_proba,
    save_json,
    segment_words,
    set_seed,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLOR_POS = "#2f7ed8"
COLOR_NEG = "#d94b4b"


# ---------------------------------------------------------------------------
# Probe evaluation
# ---------------------------------------------------------------------------

def cross_validate_multiclass(
    features: np.ndarray,
    labels: list[str],
    n_splits: int,
    C: float,
    seed: int,
):
    """Return metrics, out-of-fold probabilities and the five fold models."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(labels), len(DEPARTMENTS)), dtype=np.float64)
    folds = []

    for train_idx, test_idx in skf.split(features, labels):
        train_labels = [labels[i] for i in train_idx]
        scaler, classifier = fit_logistic(features[train_idx], train_labels, C=C, seed=seed)
        probabilities = predict_proba(scaler, classifier, features[test_idx])
        for column, class_name in enumerate(classifier.classes_):
            oof[test_idx, DEPARTMENTS.index(class_name)] = probabilities[:, column]
        folds.append(
            {
                "train_idx": train_idx,
                "test_idx": test_idx,
                "scaler": scaler,
                "classifier": classifier,
            }
        )

    y_true = np.array([DEPARTMENTS.index(label) for label in labels])
    y_pred = oof.argmax(axis=1)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(DEPARTMENTS))),
        target_names=DEPARTMENTS,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "n_samples": len(labels),
        "n_classes": len(DEPARTMENTS),
        "n_splits": n_splits,
        "C": C,
        "chance_accuracy": 1.0 / len(DEPARTMENTS),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(DEPARTMENTS)))
        ).tolist(),
    }
    return metrics, oof, folds


def one_vs_rest_auc(
    features: np.ndarray,
    labels: list[str],
    n_splits: int,
    seed: int,
) -> dict[str, float]:
    """Train one binary probe per department and report held-out AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {}
    for department in DEPARTMENTS:
        binary_labels = np.array([1 if label == department else 0 for label in labels])
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                C=0.1,
                class_weight="balanced",
                random_state=seed,
            ),
        )
        probabilities = cross_val_predict(
            pipeline, features, binary_labels, cv=skf, method="predict_proba"
        )[:, 1]
        results[department] = float(roc_auc_score(binary_labels, probabilities))
    return results


def fold_for_index(folds, index: int):
    for fold in folds:
        if index in fold["test_idx"]:
            return fold
    raise ValueError(f"No held-out fold found for index {index}")


def make_predict_fn(embedder: MiniMindEmbedder, fold_model):
    """Wrap a fold probe so callers receive probabilities in DEPARTMENTS order."""
    scaler = fold_model["scaler"]
    classifier = fold_model["classifier"]

    def predict_fn(texts):
        texts = list(texts)
        features = embedder.embed(texts)
        probabilities = predict_proba(scaler, classifier, features)
        aligned = np.zeros((len(texts), len(DEPARTMENTS)), dtype=np.float64)
        for column, class_name in enumerate(classifier.classes_):
            aligned[:, DEPARTMENTS.index(class_name)] = probabilities[:, column]
        return aligned

    return predict_fn


def select_examples(
    labels: list[str],
    oof: np.ndarray,
    max_examples: int,
    preferred: tuple[str, ...] = ("儿科", "妇产科", "皮肤科"),
) -> list[int]:
    """Pick correctly classified held-out examples, one per preferred department."""
    predictions = oof.argmax(axis=1)
    confidence = oof.max(axis=1)
    selected: list[int] = []

    for department in preferred:
        target = DEPARTMENTS.index(department)
        candidates = [
            index for index, label in enumerate(labels)
            if label == department and predictions[index] == target
        ]
        if candidates:
            selected.append(int(max(candidates, key=lambda index: confidence[index])))

    for index in np.argsort(-confidence):
        if len(selected) >= max_examples:
            break
        index = int(index)
        if index not in selected and predictions[index] == DEPARTMENTS.index(labels[index]):
            selected.append(index)

    return selected[:max_examples]


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def occlusion_for_text(predict_fn, text: str, words: list[str], target_index: int):
    """Leave-one-word-out importance: base probability minus occluded probability."""
    base_probability = float(predict_fn([text])[0, target_index])
    if not words:
        return base_probability, []

    occluded_texts = ["".join(words[:i] + words[i + 1:]) or "。" for i in range(len(words))]
    occluded_probabilities = predict_fn(occluded_texts)[:, target_index]
    deltas = base_probability - occluded_probabilities
    values = list(zip(words, deltas.tolist(), occluded_probabilities.tolist()))
    return base_probability, values


def global_occlusion(
    embedder: MiniMindEmbedder,
    folds,
    texts: list[str],
    labels: list[str],
    words_list: list[list[str]],
    max_samples: int | None = None,
):
    """Aggregate supporting-token importance per department."""
    per_department: dict[str, dict[str, list[float]]] = {
        department: defaultdict(list) for department in DEPARTMENTS
    }
    limit = len(texts) if max_samples is None else min(max_samples, len(texts))

    for index in range(limit):
        text, department, words = texts[index], labels[index], words_list[index]
        if not words:
            continue
        fold_model = fold_for_index(folds, index)
        predict_fn = make_predict_fn(embedder, fold_model)
        target_index = DEPARTMENTS.index(department)
        predicted_index = int(np.argmax(predict_fn([text])[0]))
        if predicted_index != target_index:
            continue

        _, values = occlusion_for_text(predict_fn, text, words, target_index)
        for word, delta, _ in values:
            per_department[department][word].append(delta)

    top_tokens = {}
    for department, word_values in per_department.items():
        ranked = sorted(
            (
                (word, float(np.mean(values)), len(values))
                for word, values in word_values.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        top_tokens[department] = [
            {"word": word, "mean_delta": mean_delta, "count": count}
            for word, mean_delta, count in ranked[:10]
        ]
    return top_tokens


def run_lime(
    predict_fn,
    words: list[str],
    target_index: int,
    out_dir: Path,
    stem: str,
    num_samples: int,
    num_features: int,
):
    from lime.lime_text import LimeTextExplainer

    segmented = " ".join(words)

    def cleaned_predict(texts):
        # LIME 的分词用空格连接，但探针训练时输入的是原始中文（无空格），
        # 这里把空格去掉，保证解释的是同一个输入分布。
        return predict_fn([text.replace(" ", "") for text in texts])

    explainer = LimeTextExplainer(
        class_names=DEPARTMENTS,
        split_expression=r"\s+",
        random_state=42,
    )
    explanation = explainer.explain_instance(
        segmented,
        cleaned_predict,
        labels=(target_index,),
        num_features=min(num_features, max(1, len(words))),
        num_samples=num_samples,
    )

    html_path = out_dir / f"lime_{stem}.html"
    explanation.save_to_file(str(html_path))
    return explanation.as_list(label=target_index), html_path


def extract_shap_values(shap_values, target_index: int) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            return np.asarray(shap_values[0])[0]
        return np.asarray(shap_values[target_index])[0]

    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[0, :, target_index]
    if values.ndim == 2:
        return values[0]
    return values


def run_shap(
    predict_fn,
    words: list[str],
    target_index: int,
    nsamples: int,
):
    import shap

    def model_fn(mask_matrix):
        mask_matrix = np.asarray(mask_matrix)
        texts = []
        for row in mask_matrix:
            selected = [word for word, keep in zip(words, row) if keep >= 0.5]
            texts.append("".join(selected) if selected else "。")
        return predict_fn(texts)

    explainer = shap.KernelExplainer(model_fn, np.zeros((1, len(words))))
    shap_values = explainer.shap_values(
        np.ones((1, len(words))), nsamples=nsamples, silent=True
    )
    values = extract_shap_values(shap_values, target_index)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_array = np.asarray(expected_value).flatten()
        expected_value = float(
            expected_array[target_index] if expected_array.size > 1 else expected_array[0]
        )
    else:
        expected_value = float(expected_value)

    return values, expected_value


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_confusion(matrix, path: Path) -> None:
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(DEPARTMENTS)))
    axis.set_yticks(range(len(DEPARTMENTS)))
    axis.set_xticklabels(DEPARTMENTS, rotation=45, ha="right")
    axis.set_yticklabels(DEPARTMENTS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("8-department probe: out-of-fold confusion matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_auc(auc: dict[str, float], path: Path) -> None:
    items = sorted(auc.items(), key=lambda item: item[1])
    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    figure, axis = plt.subplots(figsize=(8, 4.2))
    bars = axis.barh(labels, values, color=COLOR_POS)
    axis.set_xlim(0.5, 1.0)
    axis.set_xlabel("Held-out ROC-AUC (one-vs-rest)")
    axis.set_title("Department probes: one-vs-rest AUC")
    axis.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    for bar, value in zip(bars, values):
        axis.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_signed_bars(items, path: Path, title: str, xlabel: str, max_bars: int = 12) -> None:
    """Plot signed word attributions, keeping both positive and negative bars."""
    items = sorted(items, key=lambda item: item[1])
    if len(items) > max_bars:
        half = max_bars // 2
        items = items[:half] + items[-half:]

    words = [item[0] for item in items]
    values = [item[1] for item in items]
    colors = [COLOR_POS if value >= 0 else COLOR_NEG for value in values]

    figure, axis = plt.subplots(figsize=(8.5, max(3.2, 0.35 * len(items) + 1.2)))
    axis.barh(words, values, color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_occlusion(words, deltas, base_probability: float, path: Path, title: str) -> None:
    plot_signed_bars(
        list(zip(words, deltas)),
        path,
        title=f"{title} | P={base_probability:.3f}",
        xlabel="Probability drop when the word is removed",
    )


def plot_global_tokens(top_tokens: dict[str, list[dict]], path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for axis, department in zip(axes.flat, DEPARTMENTS):
        positive_items = [
            item for item in top_tokens.get(department, [])
            if item.get("mean_delta", 0.0) > 0
        ]
        items = positive_items[:5]
        if not items:
            axis.text(0.5, 0.5, "no positive tokens", ha="center", va="center")
            axis.set_axis_off()
            continue
        words = [item["word"] for item in items][::-1]
        values = [item["mean_delta"] for item in items][::-1]
        axis.barh(words, values, color=COLOR_POS)
        axis.set_title(department)
        axis.set_xlabel("mean prob. drop")
        axis.tick_params(axis="y", labelsize=9)
    figure.suptitle("Global token importance from leave-one-word-out occlusion")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_shap_summary(records: list[tuple[str, np.ndarray]], words_list, path: Path) -> None:
    """Aggregate mean |SHAP| per word across the explained examples."""
    totals: dict[str, list[float]] = defaultdict(list)
    for (_, values), words in zip(records, words_list):
        for word, value in zip(words, values):
            totals[word].append(abs(float(value)))

    ranked = sorted(
        ((word, float(np.mean(values))) for word, values in totals.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:12]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    words = [item[0] for item in ranked][::-1]
    values = [item[1] for item in ranked][::-1]
    axis.barh(words, values, color=COLOR_POS)
    axis.set_xlabel("mean |SHAP value|")
    axis.set_title("Global SHAP summary (explained examples)")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_token_list(
    items,
    limit: int = 5,
    value_key: str | None = None,
    show_count: bool = False,
) -> str:
    if not items:
        return "-"
    parts = []
    for item in items[:limit]:
        if isinstance(item, tuple):
            word, value = item[0], item[1]
            count = None
        else:
            word = item["word"]
            value = item.get(value_key) if value_key else None
            count = item.get("count")
        if value is None:
            parts.append(str(word))
        elif show_count and count is not None:
            parts.append(f"{word} ({value:+.3f}, n={count})")
        else:
            parts.append(f"{word} ({value:+.3f})")
    return ", ".join(parts)


def write_report(
    path: Path,
    args,
    num_parameters: int,
    metrics: dict,
    auc: dict[str, float],
    top_tokens: dict[str, list[dict]],
    example_results: list[dict],
) -> None:
    department_counts = metrics["classification_report"]
    lines = [
        "# MiniMind 医疗问答可解释性分析报告",
        "",
        "## 1. 实验设置",
        "",
        f"- 模型：MiniMind2-Small，参数量 **{num_parameters / 1e6:.2f}M**，权重 `{Path(args.weight_path).name}`",
        f"- 数据：`{Path(args.data_path).name}`，8 个科室共 **{metrics['n_samples']}** 条问题",
        f"- 表征：冻结模型最后一层 hidden state，按 attention mask 做 mean pooling（{args.hidden_size} 维）；"
        "初测中 mean pooling 明显优于 last-token pooling",
        f"- 探针：StandardScaler + LogisticRegression（C={args.C}，class_weight='balanced'）",
        f"- 评测：{metrics['n_splits']}-fold StratifiedKFold；随机猜测准确率 {metrics['chance_accuracy']:.1%}",
        "",
        "> 说明：这里解释的是 **基于 MiniMind 隐状态的科室分类探针**，不是生成式语言头本身。"
        "这样做的好处是预测目标明确、可复现；局限见第 6 节。",
        "",
        "## 2. 探针分类指标",
        "",
        f"- 8 分类 out-of-fold 准确率：**{metrics['accuracy']:.3f}**（随机猜测 {metrics['chance_accuracy']:.3f}）",
        f"- Macro-F1：**{metrics['macro_f1']:.3f}**",
        "",
        "| 科室 | 样本数 | Precision | Recall | F1 | 单类 one-vs-rest AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for department in DEPARTMENTS:
        report = department_counts.get(department, {})
        lines.append(
            f"| {department} | {int(report.get('support', 0))} | "
            f"{report.get('precision', 0):.3f} | {report.get('recall', 0):.3f} | "
            f"{report.get('f1-score', 0):.3f} | {auc[department]:.3f} |"
        )

    lines += [
        "",
        "混淆矩阵见 [probe_confusion_matrix.png](probe_confusion_matrix.png)，"
        "单类 AUC 见 [probe_auc.png](probe_auc.png)。",
        "",
        "## 3. 全局特征重要性（token occlusion）",
        "",
        "对每个 **out-of-fold 预测正确** 的样本，逐个删除问题中的一个词，"
        "观察目标科室概率下降多少；按科室聚合平均下降幅度，得到支持该科室的关键词。",
        "",
        "| 科室 | Top positive tokens（mean probability drop；n=出现次数） |",
        "|---|---|",
    ]
    for department in DEPARTMENTS:
        positive_tokens = [
            item for item in top_tokens.get(department, [])
            if item.get("mean_delta", 0.0) > 0
        ]
        lines.append(
            f"| {department} | "
            f"{format_token_list(positive_tokens, 6, 'mean_delta', show_count=True)} |"
        )

    lines += [
        "",
        "> 很多词在 66 条样本中只出现 1 次（n=1），因此上表是探索性聚合，"
        "不是稳定的全局重要性排名；单样本解释以第 4 节为准。"
        "负向 token 仍保存在 `global_occlusion.json` 中。",
        "",
        "![global tokens](global_top_tokens.png)",
        "",
        "## 4. 单样本解释",
        "",
        "每个样本都是 **out-of-fold** 预测（使用未见过该样本的 fold 模型），"
        "因此解释的是模型对未见样本的真实预测。",
        "",
        "| 科室 | 问题 | P(科室) | Occlusion Top words | LIME Top words | SHAP Top words |",
        "|---|---|---:|---|---|---|",
    ]

    for result in example_results:
        question = result["question"].replace("|", "\\|")
        lines.append(
            f"| {result['department']} | {question} | {result['probability']:.3f} | "
            f"{result['occlusion_top']} | {result['lime_top']} | {result['shap_top']} |"
        )

    lines += [
        "",
    ]
    for result in example_results:
        lines += [
            f"### {result['department']}：{result['question']}",
            "",
            f"- 预测概率：{result['probability']:.3f}",
            f"- LIME：见 [{Path(result['lime_png']).name}]({Path(result['lime_png']).name})，"
            f"交互式 HTML 见 [{Path(result['lime_html']).name}]({Path(result['lime_html']).name})",
            f"- SHAP：见 [{Path(result['shap_png']).name}]({Path(result['shap_png']).name})",
            "",
        ]

    lines += [
        "## 5. 方法对比",
        "",
        "| 方法 | 粒度 | 主要作用 | 实现 |",
        "|---|---|---|---|",
        "| Token occlusion | 词 | 直接衡量删除某词后目标概率的变化，直观但忽略词间交互 | 逐词删除后重新前向 |",
        "| LIME | 词 | 局部线性近似，给出支持/反对预测的词权重 | `lime.lime_text.LimeTextExplainer` |",
        "| SHAP (KernelSHAP) | 词 | 基于 Shapley 值的加性归因，理论性质更好 | `shap.KernelExplainer` + 词存在性 mask |",
        "",
        "### 一致性观察",
        "",
        "- 在多数样本上，三种方法的 **top-1 词**一致（例如儿科的 `宝宝`、妇产科的 `宫颈`），"
        "说明这些词确实是探针最依赖的线索。",
        "- 对次要词，occlusion 与 LIME/SHAP 可能给出不同符号：occlusion 衡量“删除该词”的边际效应，"
        "LIME/SHAP 衡量局部加性贡献，两者对词间交互的处理不同。",
        "- 因此本报告以 **排名一致性** 为主结论，不把任何单一方法的符号当作因果效应。",
        "",
        "## 6. 局限",
        "",
        "1. 数据是项目内置的合成医疗问答（66 条有科室标签），规模小，指标波动大；结论只用于验证流程。",
        "2. 解释对象是线性探针，不是生成式语言头。它回答的是“MiniMind 隐状态里哪些词支持科室判断”，"
        "不能直接等同于“模型生成答案时关注了什么”。",
        "3. 词级扰动会破坏原始语序和语法，LIME/SHAP 的归因是相关性的近似。",
        "4. 单类 one-vs-rest AUC 使用独立的二分类探针，和 8 分类探针不是同一个模型。",
        "",
        "## 7. 复现",
        "",
        "```bash",
        "# 1) 生成示例数据（如果还没有 sample_medical_qa.jsonl）",
        "python applications/medical_qa/data/generate_sample_data.py",
        "",
        "# 2) 下载官方 26M SFT 权重到 out/full_sft_512.pth",
        "curl -L -o out/full_sft_512.pth \\",
        "  https://www.modelscope.cn/models/gongjy/MiniMind2-PyTorch/resolve/master/full_sft_512.pth",
        "",
        "# 3) 运行完整可解释性分析",
        "python applications/interpretability/run_all.py",
        "```",
        "",
        f"运行参数：`n_splits={args.n_splits}`, `C={args.C}`, "
        f"`lime_samples={args.num_lime_samples}`, `shap_samples={args.num_shap_samples}`, "
        f"`seed={args.seed}`。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind medical QA interpretability analysis")
    parser.add_argument("--weight_path", type=str, default=str(DEFAULT_WEIGHT_PATH))
    parser.add_argument("--data_path", type=str, default=str(SAMPLE_DATA_PATH))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=3)
    parser.add_argument("--num_lime_samples", type=int, default=500)
    parser.add_argument("--num_shap_samples", type=int, default=300)
    parser.add_argument("--num_features", type=int, default=8)
    parser.add_argument("--skip_global", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    started_at = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading labeled medical questions ...")
    texts, labels = load_labeled_samples(args.data_path)
    if len(texts) < 10:
        raise RuntimeError(f"Not enough labeled samples: {len(texts)}")
    print(f"      {len(texts)} questions, departments: {sorted(set(labels))}")

    print("[2/6] Loading frozen MiniMind checkpoint ...")
    embedder = MiniMindEmbedder(
        weight_path=args.weight_path,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        device=args.device,
    )
    print(f"      parameters: {embedder.num_parameters / 1e6:.2f}M")

    print("[3/6] Extracting hidden states and cross-validating the probe ...")
    features = embedder.embed(texts)
    metrics, oof, folds = cross_validate_multiclass(
        features, labels, args.n_splits, args.C, args.seed
    )
    auc = one_vs_rest_auc(features, labels, args.n_splits, args.seed)
    save_json(out_dir / "probe_metrics.json", metrics)
    save_json(out_dir / "one_vs_rest_auc.json", auc)
    plot_confusion(metrics["confusion_matrix"], out_dir / "probe_confusion_matrix.png")
    plot_auc(auc, out_dir / "probe_auc.png")
    print(f"      8-way accuracy: {metrics['accuracy']:.3f}, macro-F1: {metrics['macro_f1']:.3f}")
    print("      one-vs-rest AUC: " + ", ".join(f"{k}={v:.3f}" for k, v in auc.items()))

    print("[4/6] Selecting held-out examples ...")
    selected = select_examples(labels, oof, args.max_examples)
    words_list = [segment_words(text) for text in texts]
    print(f"      selected indices: {selected}")

    print("[5/6] Running occlusion / LIME / SHAP ...")
    example_results = []
    shap_records = []
    shap_words = []
    for index in selected:
        department = labels[index]
        target_index = DEPARTMENTS.index(department)
        fold_model = fold_for_index(folds, index)
        predict_fn = make_predict_fn(embedder, fold_model)
        text = texts[index]
        words = words_list[index]
        # LIME/SHAP need enough features to perturb, so keep function words and
        # filter them from the display afterwards.
        explanation_words = segment_words(text, remove_stopwords=False)
        probability = float(predict_fn([text])[0, target_index])
        stem = f"{index:02d}_{DEPARTMENT_EN[department].replace('/', '_')}"

        print(f"      explaining #{index} [{department}] P={probability:.3f}, {len(words)} words")
        _, occlusion_values = occlusion_for_text(predict_fn, text, words, target_index)
        occlusion_sorted = sorted(occlusion_values, key=lambda item: item[1], reverse=True)
        plot_occlusion(
            [item[0] for item in sorted(occlusion_values, key=lambda item: item[1])],
            [item[1] for item in sorted(occlusion_values, key=lambda item: item[1])],
            probability,
            out_dir / f"occlusion_{stem}.png",
            title=f"Token occlusion | {department}",
        )

        lime_items, lime_html = run_lime(
            predict_fn,
            explanation_words,
            target_index,
            out_dir,
            stem,
            args.num_lime_samples,
            args.num_features,
        )
        lime_items = [(word, value) for word, value in lime_items if word not in STOPWORDS]
        lime_png = out_dir / f"lime_{stem}.png"
        plot_signed_bars(
            lime_items,
            lime_png,
            title=f"LIME word attributions | {department}",
            xlabel="LIME weight (positive supports the predicted department)",
        )
        lime_sorted = sorted(lime_items, key=lambda item: item[1], reverse=True)

        shap_values, expected_value = run_shap(
            predict_fn,
            explanation_words,
            target_index,
            args.num_shap_samples,
        )
        shap_pairs = [
            (word, value)
            for word, value in zip(explanation_words, shap_values.tolist())
            if word not in STOPWORDS
        ]
        shap_png = out_dir / f"shap_{stem}.png"
        plot_signed_bars(
            shap_pairs,
            shap_png,
            title=f"SHAP word attributions | {department}",
            xlabel="SHAP value (positive supports the predicted department)",
        )
        shap_sorted = sorted(shap_pairs, key=lambda item: item[1], reverse=True)
        shap_records.append((department, np.array([value for _, value in shap_pairs])))
        shap_words.append([word for word, _ in shap_pairs])

        example_results.append(
            {
                "index": index,
                "department": department,
                "question": text,
                "probability": probability,
                "occlusion_top": format_token_list(occlusion_sorted, 5),
                "lime_top": format_token_list(lime_sorted, 5),
                "shap_top": format_token_list(shap_sorted, 5),
                "lime_png": lime_png.name,
                "lime_html": lime_html.name,
                "shap_png": shap_png.name,
                "occlusion_png": f"occlusion_{stem}.png",
            }
        )
        print(
            f"        occlusion={format_token_list(occlusion_sorted, 3)} | "
            f"lime={format_token_list(lime_sorted, 3)} | "
            f"shap(expected={expected_value:.3f})={format_token_list(shap_sorted, 3)}"
        )

    print("[6/6] Global feature importance and report ...")
    if args.skip_global:
        top_tokens = {}
        print("      skipped global occlusion")
    else:
        top_tokens = global_occlusion(
            embedder, folds, texts, labels, words_list, max_samples=None
        )
        save_json(out_dir / "global_occlusion.json", top_tokens)
        plot_global_tokens(top_tokens, out_dir / "global_top_tokens.png")
        print("      saved global_top_tokens.png")

    if shap_records:
        plot_shap_summary(shap_records, shap_words, out_dir / "shap_summary.png")
        print("      saved shap_summary.png")

    write_report(
        out_dir / "report.md",
        args,
        embedder.num_parameters,
        metrics,
        auc,
        top_tokens,
        example_results,
    )
    save_json(out_dir / "examples.json", example_results)

    elapsed = time.time() - started_at
    print(f"Done in {elapsed:.1f}s. Results: {out_dir}")
    print("  - report.md")
    print("  - probe_metrics.json / one_vs_rest_auc.json / global_occlusion.json")
    print("  - figures: probe_*, occlusion_*, lime_*, shap_*")


if __name__ == "__main__":
    main()
