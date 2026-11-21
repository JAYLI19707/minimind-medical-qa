"""
数据准备脚本：拆分训练集/评测集 + 生成数据统计报告

功能：
1. 加载 jsonl 格式的医疗 QA 数据
2. 按比例拆分为训练集和评测集
3. 生成数据统计报告（总条数、科室分布、平均问答长度等）
4. 保存拆分后的数据文件

用法：
  python prepare_data.py                                    # 使用默认参数
  python prepare_data.py --data_path sample_medical_qa.jsonl --train_ratio 0.8 --output_dir .
  python prepare_data.py --data_path ../../dataset/lora_medical.jsonl  # 使用项目中的实际数据
"""

import json
import os
import argparse
import random
from collections import Counter


def load_jsonl(file_path):
    """从 jsonl 文件加载数据，返回数据列表"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 验证基本格式
                if "conversations" not in data:
                    print(f"[警告] 第 {line_num} 行缺少 conversations 字段，已跳过")
                    continue
                if len(data["conversations"]) < 2:
                    print(f"[警告] 第 {line_num} 行对话轮次不足，已跳过")
                    continue
                samples.append(data)
            except json.JSONDecodeError as e:
                print(f"[警告] 第 {line_num} 行 JSON 解析失败：{e}，已跳过")
    return samples


def extract_department(sample):
    """从样本中提取科室信息（如果有的话）"""
    # 优先使用显式标注的 department 字段
    if "department" in sample:
        return sample["department"]
    # 否则尝试从问题内容推断
    user_content = ""
    for turn in sample["conversations"]:
        if turn["role"] == "user":
            user_content = turn["content"]
            break
    # 简单关键词匹配
    dept_keywords = {
        "内科": ["心脏", "血压", "血糖", "糖尿病", "高血压", "胃", "肝", "肺", "咳嗽", "血脂", "幽门", "甲状腺", "贫血", "感冒"],
        "外科": ["骨折", "扭伤", "手术", "阑尾", "痔疮", "腰椎", "颈", "割伤", "烫伤", "静脉曲张"],
        "儿科": ["宝宝", "孩子", "小孩", "辅食", "发烧", "湿疹", "磨牙", "水痘"],
        "妇产科": ["月经", "怀孕", "白带", "宫颈", "痛经", "母乳", "产后"],
        "皮肤科": ["痘痘", "黑痣", "湿疹", "荨麻疹", "鸡眼", "带状疱疹", "脱发"],
        "眼科": ["眼睛", "近视", "干眼", "飞蚊", "红眼", "白内障"],
        "口腔科": ["牙齿", "牙龈", "智齿", "蛀牙", "口腔溃疡", "刷牙"],
        "中医科": ["中医", "气虚", "阳虚", "阴虚", "艾灸", "湿气", "气虚", "血虚"],
    }
    for dept, keywords in dept_keywords.items():
        for kw in keywords:
            if kw in user_content:
                return dept
    return "未分类"


def compute_statistics(samples):
    """
    计算数据集的统计信息，返回统计字典。
    """
    stats = {
        "total": len(samples),
        "departments": Counter(),
        "total_q_len": 0,
        "total_a_len": 0,
        "min_q_len": float('inf'),
        "max_q_len": 0,
        "min_a_len": float('inf'),
        "max_a_len": 0,
    }

    for sample in samples:
        # 科室统计
        dept = extract_department(sample)
        stats["departments"][dept] += 1

        # 问答长度统计
        for turn in sample["conversations"]:
            content_len = len(turn["content"])
            if turn["role"] == "user":
                stats["total_q_len"] += content_len
                stats["min_q_len"] = min(stats["min_q_len"], content_len)
                stats["max_q_len"] = max(stats["max_q_len"], content_len)
            elif turn["role"] == "assistant":
                stats["total_a_len"] += content_len
                stats["min_a_len"] = min(stats["min_a_len"], content_len)
                stats["max_a_len"] = max(stats["max_a_len"], content_len)

    if stats["total"] > 0:
        stats["avg_q_len"] = stats["total_q_len"] / stats["total"]
        stats["avg_a_len"] = stats["total_a_len"] / stats["total"]
    else:
        stats["avg_q_len"] = 0
        stats["avg_a_len"] = 0

    return stats


def print_statistics(stats, title="数据统计"):
    """格式化打印统计信息"""
    n = 60
    print("\n" + "=" * n)
    print(f"  {title}")
    print("=" * n)
    print(f"  总条数：{stats['total']}")
    print(f"  平均问题长度：{stats['avg_q_len']:.1f} 字符  "
          f"(最短: {stats['min_q_len']}, 最长: {stats['max_q_len']})")
    print(f"  平均回答长度：{stats['avg_a_len']:.1f} 字符  "
          f"(最短: {stats['min_a_len']}, 最长: {stats['max_a_len']})")

    print(f"\n  {'科室':<10} {'数量':>6} {'占比':>8}")
    print(f"  {'-'*28}")
    for dept, count in stats["departments"].most_common():
        ratio = count / stats["total"] * 100
        print(f"  {dept:<10} {count:>6} {ratio:>7.1f}%")
    print("=" * n)


def save_jsonl(samples, file_path):
    """将数据列表保存为 jsonl 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def split_and_save(samples, train_ratio, output_dir):
    """
    将数据按比例拆分为训练集和评测集，保存并打印统计信息。
    """
    random.seed(42)  # 保证可复现
    shuffled = samples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    eval_data = shuffled[split_idx:]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存文件
    train_path = os.path.join(output_dir, "train.jsonl")
    eval_path = os.path.join(output_dir, "eval.jsonl")
    save_jsonl(train_data, train_path)
    save_jsonl(eval_data, eval_path)

    # 同时保存评测数据的 question/answer 格式（供 medical_eval.py 使用）
    eval_qa_path = os.path.join(output_dir, "eval_qa.json")
    eval_qa = []
    for sample in eval_data:
        q, a = "", ""
        for turn in sample["conversations"]:
            if turn["role"] == "user":
                q = turn["content"]
            elif turn["role"] == "assistant":
                a = turn["content"]
        if q and a:
            eval_qa.append({"question": q, "answer": a})
    with open(eval_qa_path, 'w', encoding='utf-8') as f:
        json.dump(eval_qa, f, ensure_ascii=False, indent=2)
    print(f"  评测数据(JSON): {len(eval_qa)} 条 -> {eval_qa_path}")

    print(f"\n数据拆分完成：")
    print(f"  训练集：{len(train_data)} 条 -> {train_path}")
    print(f"  评测集：{len(eval_data)} 条 -> {eval_path}")

    # 分别打印训练集和评测集的统计信息
    train_stats = compute_statistics(train_data)
    eval_stats = compute_statistics(eval_data)
    print_statistics(train_stats, title=f"训练集统计 ({train_ratio*100:.0f}%)")
    print_statistics(eval_stats, title=f"评测集统计 ({(1-train_ratio)*100:.0f}%)")

    return train_data, eval_data


def main():
    parser = argparse.ArgumentParser(
        description="医疗 QA 数据准备脚本：拆分训练/评测集 + 统计报告"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="输入数据文件路径（默认为当前目录下的 sample_medical_qa.jsonl）"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集占比，默认 0.8（即 80%% 训练 / 20%% 评测）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认为数据文件所在目录）"
    )
    args = parser.parse_args()

    # 确定默认路径（相对于脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.data_path is None:
        args.data_path = os.path.join(script_dir, "sample_medical_qa.jsonl")
    # 将相对路径转为绝对路径
    args.data_path = os.path.abspath(args.data_path)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.data_path)
    args.output_dir = os.path.abspath(args.output_dir)

    # 参数校验
    if not os.path.exists(args.data_path):
        print(f"[错误] 数据文件不存在：{args.data_path}")
        print("请先运行 generate_sample_data.py 生成示例数据，或通过 --data_path 指定正确路径。")
        return

    if not (0.5 <= args.train_ratio <= 0.95):
        print(f"[错误] 训练集比例应在 0.5-0.95 之间，当前值：{args.train_ratio}")
        return

    # 加载数据
    print(f"正在加载数据：{args.data_path}")
    samples = load_jsonl(args.data_path)
    if not samples:
        print("[错误] 未加载到有效数据，请检查文件内容和格式。")
        return
    print(f"成功加载 {len(samples)} 条有效数据")

    # 打印总体统计
    total_stats = compute_statistics(samples)
    print_statistics(total_stats, title="全量数据统计")

    # 拆分并保存
    split_and_save(samples, args.train_ratio, args.output_dir)

    print("\n数据准备完成！")


if __name__ == "__main__":
    main()
