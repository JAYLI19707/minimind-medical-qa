import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

# 设置环境变量，防止 tokenizer 并行化导致的死锁或警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """
    预训练数据集类 (Pretrain Phase)
    目标：让模型学习基础的语言能力（词语接龙）。
    输入：大段的文本。
    输出：预测下一个 Token。
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        """从 jsonl 文件加载数据"""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 1. 构建输入文本
        # 将文本转换为 token id 序列，并进行 padding 和 truncation
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze() # 去掉 batch 维度
        
        # 2. 生成 loss_mask
        # 对于 padding 的部分（pad_token_id），不计算 loss
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 3. 构建训练数据对 (X, Y)
        # 自回归任务核心：预测下一个 token
        # X: 输入序列（0 到 N-1）
        # Y: 标签序列（1 到 N），即 X 中每个位置对应的“下一个”词
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long) # mask 也要对齐 Y
        
        return X, Y, loss_mask


class SFTDataset(Dataset):
    """
    监督微调数据集类 (Supervised Fine-Tuning Phase)
    目标：让模型学会对话格式，能够根据 User 的问题回答 Assistant 的内容。
    核心逻辑：只计算 Assistant 回复部分的 Loss，忽略 User 提问部分的 Loss。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        # 预先计算 assistant 起始和结束的 token ids，用于后续生成 mask
        # 例如: <|im_start|>assistant
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, cs):
        """
        构建对话 Prompt
        使用 tokenizer.apply_chat_template 将 messages 列表转换为模型训练所需的字符串格式
        """
        messages = cs.copy()
        # 处理 function call (如果存在)
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def _generate_loss_mask(self, input_ids):
        """
        ⭐️ 核心函数：生成 Loss Mask
        逻辑：遍历 input_ids，找到所有 'assistant' 回复的片段，将这些片段的 mask 设为 1，其余（如 user 提问、padding）设为 0。
        这样模型训练时只会“努力”去预测 assistant 的回答，而不会去预测 user 的问题。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 找到 <|im_start|>assistant 的开始位置
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 寻找该轮回复的结束位置 <|im_end|>
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 assistant 回复部分（包括结束符）的 mask 设为 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 更新指针，继续寻找下一轮对话
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 1. 构建对话提示字符串
        prompt = self._create_chat_prompt(sample['conversations'])
        
        # 2. Tokenize 并截断/填充
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 3. 生成动态损失掩码 (只训练 assistant 的回答)
        loss_mask = self._generate_loss_mask(input_ids)

        # 4. 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} Token Loss Mask (length: {len(input_ids)}) ---")
        # for i, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        #     token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        #     token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')  # 处理换行等不可见字符
        #     print(f"Token {i:3d}: {token_id:5d} -> '{token_str:10s}' | mask: {mask}")
        # print(f"--- End of Sample {index} ---")
        # # ================================
        return X, Y, loss_mask


class DPODataset(Dataset):
    """
    DPO 数据集类 (Direct Preference Optimization)
    目标：用于 RLHF 阶段，学习人类偏好。
    输入：成对的数据 (Chosen, Rejected)。
    逻辑：分别处理 Chosen 和 Rejected 数据，同样需要 mask 掉 user 的提问。
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        
        # 加载 DPO jsonl 数据
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        
        # 2. Tokenize
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        # 3. 获取 Input IDs
        chosen_input_ids = chosen_encoding['input_ids']
        rejected_input_ids = rejected_encoding['input_ids']

        # 4. 生成 Loss Mask (Mask 掉 User 部分)
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        
        # 5. 构建 Tensor (X, Y, Mask)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        """同 SFTDataset，生成只保留 assistant 回复的 mask"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """
    RLAIF 数据集类 (Reinforcement Learning from AI Feedback)
    目标：用于 PPO/GRPO 等强化学习训练。
    输入：Prompt (问题)。
    区别：这里只需要 User 的 Prompt，模型会实时生成 Response，不需要预先的 Answer。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content'] # 暂存最后的回答作为参考
        
        # apply_chat_template 生成 prompt 时，add_generation_prompt=True 会自动添加 <|im_start|>assistant
        # 这样模型接下来的输出就是 Assistant 的回复
        return self.tokenizer.apply_chat_template(
            messages[:-1], # 去掉最后一个 assistant 的回复，只保留前面的对话
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        ), answer

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建提示 Prompt (用于让 Actor 模型生成)
        prompt, answer = self._create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }


if __name__ == "__main__":
    pass
