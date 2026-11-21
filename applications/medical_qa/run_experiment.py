"""
方案5：医疗问答域应用 - 一键实验脚本
串联数据准备、LoRA 微调、评测对比、报告生成、Demo 启动全流程。

用法：
    python run_experiment.py                          # 运行全部步骤
    python run_experiment.py --skip_train             # 跳过训练（使用已有 LoRA 权重）
    python run_experiment.py --skip_eval              # 跳过评测
    python run_experiment.py --demo                   # 仅启动 Demo
    python run_experiment.py --epochs 10 --batch_size 8 --learning_rate 5e-5
"""

import os
import sys
import argparse
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

# ============================================================================
# 路径设置：确保可以导入项目根目录下的模块
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # applications/medical_qa/
PROJECT_ROOT = SCRIPT_DIR.parent.parent               # 项目根目录
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# 常量
# ============================================================================
DEFAULT_LORA_NAME = "lora_medical"
DEFAULT_DATA_PATH = SCRIPT_DIR / "data" / "train.jsonl"
DEFAULT_EVAL_DATA = SCRIPT_DIR / "data" / "eval_qa.json"
RESULTS_DIR = SCRIPT_DIR / "results"
DEMO_SCRIPT = SCRIPT_DIR / "demo" / "medical_chat.py"
REPORT_PATH = RESULTS_DIR / "report.md"


# ============================================================================
# 工具函数
# ============================================================================
def banner(msg: str, width: int = 60):
    """打印醒目的步骤标题"""
    line = "=" * width
    print(f"\n{line}")
    print(f"  {msg}")
    print(f"{line}\n")


def check_file(path, desc: str) -> bool:
    """检查文件是否存在，不存在时打印提示"""
    if Path(path).exists():
        print(f"  [OK] {desc}: {path}")
        return True
    else:
        print(f"  [MISSING] {desc}: {path}")
        return False


def run_subprocess(cmd: list, cwd: str = None, desc: str = "") -> bool:
    """运行子进程，返回是否成功"""
    print(f"  执行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or str(PROJECT_ROOT),
            check=True,
            capture_output=False,
        )
        if result.returncode == 0:
            print(f"  [OK] {desc} 完成")
            return True
        else:
            print(f"  [FAIL] {desc} 返回码: {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] {desc} 失败，错误: {e}")
        return False
    except FileNotFoundError:
        print(f"  [FAIL] 找不到命令: {cmd[0]}")
        return False


# ============================================================================
# 步骤 1：数据准备
# ============================================================================
def step_data_prepare(args) -> bool:
    """调用 data/prepare_data.py 生成/拆分数据"""
    banner("步骤 1/4：数据准备")

    prepare_script = SCRIPT_DIR / "data" / "prepare_data.py"

    if not check_file(prepare_script, "数据准备脚本"):
        print("  [INFO] 未找到 data/prepare_data.py，跳过自动数据准备。")
        print("  [INFO] 请确保以下文件已手动准备:")
        print(f"         训练数据: {args.data_path}")
        print(f"         评测数据: {DEFAULT_EVAL_DATA}")
        return True  # 不阻塞后续步骤

    cmd = [
        sys.executable, str(prepare_script),
        "--output_dir", str(SCRIPT_DIR / "data"),
    ]
    if args.data_path:
        cmd.extend(["--data_path", str(args.data_path)])

    success = run_subprocess(cmd, desc="数据准备")
    if not success:
        print("  [提示] 数据准备失败，请检查 data/prepare_data.py 脚本。")
    return success


# ============================================================================
# 步骤 2：LoRA 微调
# ============================================================================
def step_lora_train(args) -> bool:
    """调用 trainer/train_lora.py 进行 LoRA 微调"""
    banner("步骤 2/4：LoRA 微调")

    # 检查训练数据
    data_path = Path(args.data_path) if args.data_path else DEFAULT_DATA_PATH
    if not check_file(data_path, "训练数据"):
        print(f"  [ERROR] 训练数据不存在: {data_path}")
        print("  [提示] 请先运行数据准备步骤，或通过 --data_path 指定数据路径。")
        return False

    # 检查 base 权重
    base_weight = PROJECT_ROOT / "out" / f"full_sft_512.pth"
    if not check_file(base_weight, "Base 模型权重"):
        print("  [ERROR] 未找到 full_sft 基础权重。")
        print("  [提示] 请先完成预训练和 SFT，或从其他来源获取权重。")
        return False

    # 构造训练命令：在项目根目录下运行 trainer/train_lora.py
    cmd = [
        sys.executable, os.path.join("trainer", "train_lora.py"),
        "--lora_name", DEFAULT_LORA_NAME,
        "--data_path", str(data_path),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--save_dir", "out/lora",
        "--hidden_size", "512",
        "--num_hidden_layers", "8",
        "--max_seq_len", "512",
        "--from_weight", "full_sft",
    ]

    print(f"  训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    print(f"  LoRA 名称: {DEFAULT_LORA_NAME}")
    print(f"  训练数据: {data_path}")
    print()

    success = run_subprocess(cmd, desc="LoRA 微调")
    if not success:
        print("  [提示] 训练失败。常见原因:")
        print("         1. GPU 显存不足 -> 减小 --batch_size 或 --max_seq_len")
        print("         2. 训练数据格式错误 -> 检查 JSONL 格式")
        print("         3. 缺少依赖包 -> pip install -r requirements.txt")
    return success


# ============================================================================
# 步骤 3：评测对比
# ============================================================================
def step_eval(args) -> bool:
    """调用 eval/medical_eval.py 运行 base vs lora 评测"""
    banner("步骤 3/4：评测对比")

    eval_script = SCRIPT_DIR / "eval" / "medical_eval.py"

    if not check_file(eval_script, "评测脚本"):
        print("  [INFO] 未找到 eval/medical_eval.py，跳过自动评测。")
        return True

    # 检查 LoRA 权重是否存在
    lora_weight_path = PROJECT_ROOT / "out" / "lora" / f"{DEFAULT_LORA_NAME}_512.pth"
    has_lora = check_file(lora_weight_path, "LoRA 权重")

    if not has_lora:
        print("  [WARN] LoRA 权重不存在，评测将仅运行 Base 模型。")
        print("  [提示] 请先完成 LoRA 微调步骤。")

    # 检查评测数据
    eval_data = DEFAULT_EVAL_DATA
    if not check_file(eval_data, "评测数据"):
        print("  [WARN] 评测数据不存在，尝试使用训练数据的一部分作为评测数据。")
        eval_data = Path(args.data_path) if args.data_path else DEFAULT_DATA_PATH

    cmd = [
        sys.executable, str(eval_script),
        "--eval_data", str(eval_data),
        "--tokenizer_path", "model",
        "--save_dir", "out",
        "--hidden_size", "512",
        "--num_hidden_layers", "8",
    ]
    if has_lora:
        cmd.extend(["--lora_weight", DEFAULT_LORA_NAME])

    success = run_subprocess(cmd, desc="评测对比")
    if not success:
        print("  [提示] 评测失败，请检查 eval/medical_eval.py 脚本。")
    return success


# ============================================================================
# 步骤 4：生成报告
# ============================================================================
def step_generate_report(args, results: dict) -> bool:
    """输出 markdown 格式的实验报告"""
    banner("步骤 4/4：生成实验报告")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 尝试读取评测结果文件（如果存在）
    eval_results_path = RESULTS_DIR / "eval_results.json"
    eval_data = {}
    if eval_results_path.exists():
        try:
            with open(eval_results_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            print(f"  [OK] 读取评测结果: {eval_results_path}")
        except Exception as e:
            print(f"  [WARN] 读取评测结果失败: {e}")

    report = f"""# 方案5：医疗问答域应用 - 实验报告

> 生成时间: {now}

## 实验配置

| 参数 | 值 |
|------|-----|
| 基础模型 | MiniMind (hidden_size=512, layers=8, ~26M) |
| 微调方法 | LoRA (rank=8) |
| LoRA 名称 | {DEFAULT_LORA_NAME} |
| 训练轮数 | {args.epochs} |
| Batch Size | {args.batch_size} |
| 学习率 | {args.learning_rate} |
| 训练数据 | {args.data_path or str(DEFAULT_DATA_PATH)} |
| 最大序列长度 | 512 |

## 步骤执行结果

| 步骤 | 状态 |
|------|------|
| 数据准备 | {"完成" if results.get("data_prepare", True) else "失败/跳过"} |
| LoRA 微调 | {"完成" if results.get("lora_train", True) else "失败/跳过"} |
| 评测对比 | {"完成" if results.get("eval", True) else "失败/跳过"} |

## 评测指标

"""
    if eval_data:
        report += "### Base 模型 (full_sft)\n\n"
        base = eval_data.get("base", {})
        if base:
            report += "| 指标 | 值 |\n|------|-----|\n"
            for k, v in base.items():
                report += f"| {k} | {v} |\n"
        else:
            report += "> 暂无数据\n"

        report += "\n### LoRA 微调模型\n\n"
        lora = eval_data.get("lora", {})
        if lora:
            report += "| 指标 | 值 |\n|------|-----|\n"
            for k, v in lora.items():
                report += f"| {k} | {v} |\n"
        else:
            report += "> 暂无数据\n"

        report += "\n### 对比总结\n\n"
        improvements = eval_data.get("improvements", {})
        if improvements:
            report += "| 指标 | 提升 |\n|------|------|\n"
            for k, v in improvements.items():
                report += f"| {k} | {v} |\n"
        else:
            report += "> 暂无数据\n"
    else:
        report += """
> 评测结果文件 (results/eval_results.json) 不存在。
> 请先运行评测步骤 (`python run_experiment.py`)，或手动运行:
> ```
> python eval/medical_eval.py --base_weight full_sft --lora_name lora_medical
> ```
"""

    report += f"""
## 模型文件

| 文件 | 路径 | 状态 |
|------|------|------|
| Base 权重 | out/full_sft_512.pth | {"存在" if (PROJECT_ROOT / "out" / "full_sft_512.pth").exists() else "不存在"} |
| LoRA 权重 | out/lora/{DEFAULT_LORA_NAME}_512.pth | {"存在" if (PROJECT_ROOT / "out" / "lora" / f"{DEFAULT_LORA_NAME}_512.pth").exists() else "不存在"} |

## 后续步骤

1. 如需重新训练：`python run_experiment.py`
2. 仅运行评测：`python run_experiment.py --skip_train`
3. 启动 Demo：`python run_experiment.py --demo`
4. 仅生成报告：`python run_experiment.py --skip_train --skip_eval`
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  [OK] 实验报告已生成: {REPORT_PATH}")
    return True


# ============================================================================
# 步骤 5（可选）：启动 Demo
# ============================================================================
def step_launch_demo(args) -> bool:
    """调用 Streamlit 启动医疗问答 Demo"""
    banner("启动 Demo")

    if not check_file(DEMO_SCRIPT, "Demo 脚本"):
        print(f"  [ERROR] Demo 脚本不存在: {DEMO_SCRIPT}")
        print("  [提示] 请确保 demo/medical_demo.py 已创建。")
        return False

    # 检查 streamlit 是否安装
    try:
        import streamlit  # noqa: F401
        print(f"  [OK] Streamlit 版本: {streamlit.__version__}")
    except ImportError:
        print("  [ERROR] 未安装 streamlit，请执行: pip install streamlit")
        return False

    print(f"  启动 Streamlit Demo: {DEMO_SCRIPT}")
    print("  浏览器将自动打开，地址: http://localhost:8501")
    print("  按 Ctrl+C 停止服务。\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(DEMO_SCRIPT)],
            cwd=str(SCRIPT_DIR),
            check=True,
        )
        return True
    except KeyboardInterrupt:
        print("\n  [INFO] Demo 已停止。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] Demo 启动失败: {e}")
        return False


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="方案5：医疗问答域应用 - 一键实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_experiment.py                              # 运行全部步骤
  python run_experiment.py --skip_train                 # 跳过训练
  python run_experiment.py --skip_train --skip_eval     # 仅生成报告
  python run_experiment.py --demo                       # 仅启动 Demo
  python run_experiment.py --epochs 10 --batch_size 8   # 自定义训练参数
        """,
    )

    # 跳过选项
    parser.add_argument("--skip_train", action="store_true", help="跳过 LoRA 微调步骤")
    parser.add_argument("--skip_eval", action="store_true", help="跳过评测对比步骤")
    parser.add_argument("--demo", action="store_true", help="仅启动 Demo（跳过训练和评测）")

    # 训练参数
    parser.add_argument("--data_path", type=str, default=None, help="训练数据路径（默认: data/medical_sft.jsonl）")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数（默认: 5）")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size（默认: 16）")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率（默认: 1e-4）")

    args = parser.parse_args()

    # 设置默认数据路径
    if args.data_path is None:
        args.data_path = str(DEFAULT_DATA_PATH)

    # ---- 仅 Demo 模式 ----
    if args.demo:
        step_launch_demo(args)
        return

    # ---- 完整流程 ----
    start_time = time.time()
    results = {}

    banner("方案5：医疗问答域应用 - 实验开始")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  工作目录:   {SCRIPT_DIR}")
    print(f"  训练数据:   {args.data_path}")
    print(f"  跳过训练:   {args.skip_train}")
    print(f"  跳过评测:   {args.skip_eval}")
    print()

    # 步骤 1：数据准备
    results["data_prepare"] = step_data_prepare(args)

    # 步骤 2：LoRA 微调
    if args.skip_train:
        banner("步骤 2/4：LoRA 微调 [已跳过]")
        print("  用户指定 --skip_train，跳过训练步骤。")
        lora_path = PROJECT_ROOT / "out" / "lora" / f"{DEFAULT_LORA_NAME}_512.pth"
        if check_file(lora_path, "已有 LoRA 权重"):
            print("  [INFO] 将使用已有权重进行评测。")
        else:
            print("  [WARN] LoRA 权重不存在，评测可能无法对比 Base vs LoRA。")
        results["lora_train"] = True
    else:
        results["lora_train"] = step_lora_train(args)

    # 步骤 3：评测对比
    if args.skip_eval:
        banner("步骤 3/4：评测对比 [已跳过]")
        results["eval"] = True
    else:
        results["eval"] = step_eval(args)

    # 步骤 4：生成报告
    step_generate_report(args, results)

    # ---- 完成 ----
    elapsed = time.time() - start_time
    banner("实验完成")
    print(f"  总耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
    print(f"  实验报告: {REPORT_PATH}")
    print()

    # 如果评测结果文件存在，打印摘要
    eval_results_path = RESULTS_DIR / "eval_results.json"
    if eval_results_path.exists():
        try:
            with open(eval_results_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            print("  评测结果摘要:")
            for model_name in ["base", "lora"]:
                if model_name in eval_data:
                    metrics = eval_data[model_name]
                    bleu = metrics.get("BLEU-4", metrics.get("bleu4", "N/A"))
                    rouge = metrics.get("ROUGE-L", metrics.get("rouge_l", "N/A"))
                    print(f"    {model_name:>8s}: BLEU-4={bleu}, ROUGE-L={rouge}")
        except Exception:
            pass

    print("\n  如需启动 Demo，请运行: python run_experiment.py --demo\n")


if __name__ == "__main__":
    main()
