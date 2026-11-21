"""
数据扩充脚本：解决 100 条数据太少的问题

提供三种数据扩充策略：
1. 下载项目官方 lora_medical.jsonl（34MB，推荐）
2. 从开源中文医疗数据集扩充
3. 基于已有数据做同义替换扩充

用法：
  python augment_data.py --method download     # 下载官方数据（推荐）
  python augment_data.py --method augment --target 500  # 扩充到 500 条
  python augment_data.py --method merge         # 合并所有数据源
"""

import json
import os
import argparse
import random
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# 方法 1：下载项目官方 lora_medical.jsonl
# ============================================================
def download_official_data():
    """从 HuggingFace/ModelScope 下载项目官方医疗数据集（34MB）"""
    print("=" * 60)
    print("下载项目官方 lora_medical.jsonl（34MB）")
    print("=" * 60)

    output_path = SCRIPT_DIR / "lora_medical.jsonl"

    # 方式 1：HuggingFace
    try:
        from huggingface_hub import hf_hub_download
        print("正在从 HuggingFace 下载...")
        path = hf_hub_download(
            repo_id="jingyaogong/minimind_dataset",
            filename="lora_medical.jsonl",
            local_dir=str(SCRIPT_DIR),
        )
        print(f"下载完成: {path}")

        # 统计
        count = count_jsonl(path)
        print(f"数据条数: {count}")
        return path
    except Exception as e:
        print(f"HuggingFace 下载失败: {e}")

    # 方式 2：ModelScope
    try:
        print("尝试从 ModelScope 下载...")
        from modelscope import MsDataset
        ds = MsDataset.load("gongjy/minimind_dataset", split="lora_medical")
        # 保存为 jsonl
        with open(output_path, "w", encoding="utf-8") as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        count = count_jsonl(str(output_path))
        print(f"下载完成: {output_path}")
        print(f"数据条数: {count}")
        return str(output_path)
    except Exception as e:
        print(f"ModelScope 下载失败: {e}")

    print("\n自动下载失败，请手动下载：")
    print("  HuggingFace: https://huggingface.co/datasets/jingyaogong/minimind_dataset")
    print("  ModelScope:  https://www.modelscope.cn/datasets/gongjy/minimind_dataset")
    print(f"\n下载后将 lora_medical.jsonl 放到: {SCRIPT_DIR}")
    return None


# ============================================================
# 方法 2：基于已有数据做同义替换扩充
# ============================================================

# 医疗领域同义词表（用于扩充）
SYNONYMS = {
    "头痛": ["头疼", "脑袋疼", "偏头痛"],
    "发烧": ["发热", "体温升高", "高热"],
    "咳嗽": ["干咳", "咳痰", "持续咳嗽"],
    "恶心": ["反胃", "想吐", "作呕"],
    "腹泻": ["拉肚子", "水样便", "大便稀"],
    "失眠": ["睡不着", "入睡困难", "睡眠障碍"],
    "头晕": ["眩晕", "头昏", "晕眩"],
    "胸闷": ["胸口发闷", "憋气", "呼吸不畅"],
    "皮疹": ["红疹", "皮肤起疹", "出疹子"],
    "疲劳": ["乏力", "没精神", "浑身没劲"],
    "心悸": ["心慌", "心跳加速", "心脏跳得快"],
    "关节痛": ["关节疼", "骨节疼痛", "关节酸痛"],
    "牙疼": ["牙痛", "牙齿疼痛", "牙龈肿痛"],
    "眼睛干涩": ["眼干", "眼睛发干", "眼部干涩"],
    "嗓子疼": ["咽喉痛", "喉咙痛", "咽部疼痛"],
    "胃痛": ["胃疼", "上腹痛", "胃部不适"],
    "腰痛": ["腰疼", "腰部疼痛", "腰酸背痛"],
    "水肿": ["浮肿", "肿胀", "肢体肿胀"],
    "瘙痒": ["发痒", "皮肤痒", "痒得厉害"],
    "食欲不振": ["不想吃饭", "没胃口", "食欲减退"],
    "怎么办": ["怎么处理", "如何治疗", "该怎么治"],
    "什么原因": ["怎么回事", "是什么引起的", "为什么会这样"],
    "需要注意什么": ["有什么注意事项", "应该注意哪些", "日常要注意什么"],
    "严重吗": ["有没有危险", "情况严不严重", "需要担心吗"],
    "需要手术吗": ["必须做手术吗", "能不能不手术", "手术治疗吗"],
    "吃什么药好": ["用什么药", "该服什么药", "推荐什么药物"],
}


def augment_with_synonyms(samples, target_count):
    """通过同义词替换扩充数据"""
    augmented = list(samples)  # 保留原始数据
    random.seed(42)

    while len(augmented) < target_count:
        # 随机选一条原始数据做变换
        sample = random.choice(samples)
        new_sample = json.loads(json.dumps(sample))  # 深拷贝

        # 对 user 的问题做同义替换
        for turn in new_sample["conversations"]:
            if turn["role"] == "user":
                text = turn["content"]
                for word, synonyms in SYNONYMS.items():
                    if word in text and random.random() < 0.5:
                        replacement = random.choice(synonyms)
                        text = text.replace(word, replacement, 1)
                        break  # 每条只替换一处，保持自然
                turn["content"] = text

        augmented.append(new_sample)

    random.shuffle(augmented)
    return augmented[:target_count]


# ============================================================
# 方法 3：从问题模板批量生成
# ============================================================

QUESTION_TEMPLATES = [
    "我最近总是{symptom}，是什么原因？",
    "{symptom}已经好几天了，需要去医院吗？",
    "{symptom}怎么办？吃什么药好？",
    "请问{symptom}是什么原因引起的？",
    "{symptom}反复发作，需要注意什么？",
    "{symptom}严重吗？需要做什么检查？",
    "如果出现{symptom}，应该怎么处理？",
    "{symptom}的治疗方法有哪些？",
    "{symptom}能根治吗？",
    "{symptom}跟饮食有关系吗？",
]

SYMPTOMS = [
    "头痛", "发烧", "咳嗽", "恶心", "腹泻", "失眠", "头晕",
    "胸闷", "皮疹", "疲劳", "心悸", "关节痛", "牙疼",
    "嗓子疼", "胃痛", "腰痛", "水肿", "瘙痒", "食欲不振",
    "耳鸣", "手抖", "出汗多", "口干舌燥", "便秘",
]


def generate_from_templates(count):
    """从模板批量生成问题（仅生成问题，答案需要模型生成或人工填写）"""
    samples = []
    random.seed(42)

    for _ in range(count):
        symptom = random.choice(SYMPTOMS)
        template = random.choice(QUESTION_TEMPLATES)
        question = template.format(symptom=symptom)

        # 生成一个基础回答模板
        answer = (
            f"{symptom}可能由多种原因引起，建议您到医院做相关检查明确病因。"
            f"日常生活中注意休息、均衡饮食、适当运动。"
            f"如果症状持续或加重，请及时就医。"
        )

        samples.append({
            "conversations": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })

    return samples


# ============================================================
# 合并多个数据源
# ============================================================
def merge_datasets(file_list, output_path):
    """合并多个 jsonl 文件"""
    all_samples = []
    seen_questions = set()

    for fpath in file_list:
        if not os.path.exists(fpath):
            print(f"  跳过（不存在）: {fpath}")
            continue
        count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # 去重：检查问题是否已存在
                q = ""
                for turn in data.get("conversations", []):
                    if turn["role"] == "user":
                        q = turn["content"][:50]  # 取前 50 字符做去重
                        break
                if q and q not in seen_questions:
                    seen_questions.add(q)
                    all_samples.append(data)
                    count += 1
        print(f"  加载 {count} 条: {fpath}")

    random.seed(42)
    random.shuffle(all_samples)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n合并完成: {len(all_samples)} 条 -> {output_path}")
    return all_samples


# ============================================================
# 工具函数
# ============================================================
def count_jsonl(file_path):
    """统计 jsonl 文件行数"""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def split_and_save(samples, train_ratio=0.8):
    """拆分数据并保存（覆盖 prepare_data.py 的功能）"""
    random.seed(42)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    eval_data = shuffled[split_idx:]

    train_path = SCRIPT_DIR / "train.jsonl"
    eval_path = SCRIPT_DIR / "eval.jsonl"
    eval_qa_path = SCRIPT_DIR / "eval_qa.json"

    # 保存训练集（conversations 格式）
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 保存评测集（conversations 格式）
    with open(eval_path, "w", encoding="utf-8") as f:
        for sample in eval_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 保存评测集（question/answer 格式，供 medical_eval.py 使用）
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

    with open(eval_qa_path, "w", encoding="utf-8") as f:
        json.dump(eval_qa, f, ensure_ascii=False, indent=2)

    print(f"\n数据拆分完成:")
    print(f"  训练集: {len(train_data)} 条 -> {train_path}")
    print(f"  评测集: {len(eval_data)} 条 -> {eval_path}")
    print(f"  评测JSON: {len(eval_qa)} 条 -> {eval_qa_path}")

    return train_data, eval_data


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="医疗 QA 数据扩充脚本")
    parser.add_argument("--method", type=str, default="download",
                        choices=["download", "augment", "merge", "all"],
                        help="扩充方法: download=下载官方数据, augment=同义替换扩充, merge=合并所有数据源, all=全部执行")
    parser.add_argument("--target", type=int, default=500,
                        help="扩充目标条数（默认 500）")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例（默认 0.8）")
    args = parser.parse_args()

    sample_path = str(SCRIPT_DIR / "sample_medical_qa.jsonl")
    official_path = str(SCRIPT_DIR / "lora_medical.jsonl")
    augmented_path = str(SCRIPT_DIR / "augmented_medical_qa.jsonl")

    if args.method == "download" or args.method == "all":
        download_official_data()

    if args.method == "augment" or args.method == "all":
        print("\n" + "=" * 60)
        print(f"同义替换扩充（目标: {args.target} 条）")
        print("=" * 60)

        # 加载已有数据
        source_path = official_path if os.path.exists(official_path) else sample_path
        if not os.path.exists(source_path):
            print(f"源数据不存在: {source_path}")
            print("请先运行 generate_sample_data.py 或下载官方数据")
            return

        samples = []
        with open(source_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        print(f"源数据: {len(samples)} 条 ({source_path})")

        # 扩充
        augmented = augment_with_synonyms(samples, args.target)

        with open(augmented_path, "w", encoding="utf-8") as f:
            for sample in augmented:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"扩充完成: {len(augmented)} 条 -> {augmented_path}")

    if args.method == "merge" or args.method == "all":
        print("\n" + "=" * 60)
        print("合并所有数据源")
        print("=" * 60)

        # 收集所有可用数据文件
        files_to_merge = []
        for f in [sample_path, official_path, augmented_path]:
            if os.path.exists(f):
                files_to_merge.append(f)

        if not files_to_merge:
            print("没有找到任何数据文件")
            return

        all_samples = merge_datasets(files_to_merge, official_path)

        # 拆分
        split_and_save(all_samples, args.train_ratio)

    # 如果只是 download，也做拆分
    if args.method == "download":
        if os.path.exists(official_path):
            samples = []
            with open(official_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            split_and_save(samples, args.train_ratio)

    print("\n" + "=" * 60)
    print("数据扩充完成！")
    print("=" * 60)
    print(f"\n后续步骤:")
    print(f"  1. 训练: python run_experiment.py")
    print(f"  2. 或手动: python -m trainer.train_lora --data_path data/train.jsonl ...")


if __name__ == "__main__":
    main()
