# AutoDL 运行方案5：医疗问答域应用 —— 一步一步教你跑

> 预计总耗时：1 小时 | 预计费用：1-3 元
> 前提：你有一个 AutoDL 账号（https://www.autodl.com），已充值 10 元

---

## 第一步：创建 GPU 实例

1. 登录 AutoDL -> 控制台 -> 容器实例 -> **创建新实例**
2. 配置选择：

```
GPU 数量：1 块
GPU 型号：RTX 3090（1-2 元/时，性价比最高）
         或 RTX 3080（更便宜，也够用）
镜像选择：PyTorch 2.x / CUDA 12.x / Python 3.x (ubuntu22.04)
         （不需要完全匹配版本，PyTorch 2.x 即可）
数据盘：默认 50GB 够用
```

3. 等待实例创建完成（约 1-2 分钟）
4. 创建完成后，点 **JupyterLab** 进入（比 SSH 更方便）

---

## 第二步：打开终端

进入 JupyterLab 后，点击顶部菜单栏 **File -> New -> Terminal**，或点击左上角的 **终端** 图标。

之后所有命令都在这个终端里执行。

---

## 第三步：获取项目代码

### 方式一：Git 克隆（推荐）

```bash
cd /root
git clone https://github.com/jingyaogong/minimind.git
cd minimind
```

### 方式二：手动上传

如果 Git 不可用，在 JupyterLab 左侧文件浏览器中：
1. 点左侧 `root` 文件夹
2. 把你本地的 `minimind-master` 整个文件夹拖进去上传
3. 上传完后终端执行：

```bash
cd /root/minimind-master  # 或你上传的文件夹名
```

---

## 第四步：安装依赖

依次执行以下命令（一条一条复制粘贴）：

```bash
cd /root/minimind
```

```bash
pip install -r requirements.txt
```

```bash
pip install streamlit
```

等待安装完成（约 3-5 分钟）。如果某个包装失败，重新执行一次那条命令即可。

---

## 第五步：验证 GPU 可用

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

你应该看到类似这样的输出：

```
PyTorch: 2.6.0
CUDA: True
GPU: NVIDIA GeForce RTX 3090
```

如果 `CUDA: True` 说明一切正常，可以继续。如果是 `False`，说明选的镜像没有 CUDA，需要换一个。

---

## 第六步：生成示例数据

```bash
cd /root/minimind/applications/medical_qa
```

```bash
python data/generate_sample_data.py
```

你应该看到：

```
==================================================
示例医疗 QA 数据生成完成
==================================================
总条数：100
保存路径：/root/minimind/applications/medical_qa/data/sample_medical_qa.jsonl
...
```

然后拆分训练集和评测集：

```bash
python data/prepare_data.py
```

你应该看到：

```
数据拆分完成：
  训练集：80 条 -> .../data/train.jsonl
  评测集：20 条 -> .../data/eval.jsonl
  评测数据(JSON): 20 条 -> .../data/eval_qa.json
```

---

## 第七步：检查基础模型权重

方案5 需要 `full_sft_512.pth` 作为基础模型。检查一下有没有：

```bash
ls -la /root/minimind/out/full_sft_512.pth
```

### 情况 A：文件存在

恭喜，直接跳到 **第八步**。

### 情况 B：文件不存在

你需要获取基础权重。以下是几种方法：

#### 方法 1：从 HuggingFace 下载（推荐）

```bash
pip install huggingface_hub
```

```bash
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='jingyaogong/MiniMind', filename='full_sft_512.pth', local_dir='/root/minimind/out')
print(f'下载完成: {path}')
"
```

如果 HuggingFace 访问慢，用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='jingyaogong/MiniMind', filename='full_sft_512.pth', local_dir='/root/minimind/out')
print(f'下载完成: {path}')
"
```

#### 方法 2：从 ModelScope 下载（国内更快）

```bash
pip install modelscope
```

```bash
python -c "
from modelscope import snapshot_download
snapshot_download('AI-ModelScope/MiniMind', local_dir='/root/minimind/out')
print('下载完成')
"
```

#### 方法 3：自己训练（耗时较长，需要预训练数据）

如果你有预训练数据集（`pretrain_hq.jsonl`），可以自己训练：

```bash
cd /root/minimind

# 先检查数据是否存在
ls dataset/pretrain_hq.jsonl

# 如果数据存在，开始训练
python trainer/train_pretrain.py --epochs 2 --hidden_size 512 --num_hidden_layers 8
python trainer/train_full_sft.py --epochs 2 --hidden_size 512 --num_hidden_layers 8
```

#### 方法 4：跳过训练，只体验 Demo 和数据

如果没有权重也不想训练，可以直接跳到 **第十步** 启动 Demo（会进入 mock 演示模式，展示界面但不跑真实模型）。

---

## 第八步：LoRA 微调训练

确保你在项目根目录：

```bash
cd /root/minimind
```

执行 LoRA 微调（用医疗 QA 数据训练，约 5-10 分钟）：

```bash
python -m trainer.train_lora \
    --lora_name lora_medical \
    --data_path applications/medical_qa/data/train.jsonl \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --max_seq_len 512 \
    --from_weight full_sft
```

你应该看到训练日志在打印 loss：

```
Epoch:[1/5](1/5) loss:2.345678 lr:... epoch_Time:...min:
Epoch:[1/5](2/5) loss:1.987654 lr:... epoch_Time:...min:
...
```

训练完成后，LoRA 权重保存在：

```
/root/minimind/out/lora/lora_medical_512.pth
```

验证一下：

```bash
ls -la /root/minimind/out/lora/lora_medical_512.pth
```

---

## 第九步：运行评测

```bash
cd /root/minimind
```

```bash
python applications/medical_qa/eval/medical_eval.py \
    --eval_data applications/medical_qa/data/eval_qa.json \
    --tokenizer_path model \
    --save_dir out \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --lora_weight lora_medical \
    --device cuda
```

评测会依次输出：
1. **准确率评测**（BLEU、ROUGE-L）
2. **生成质量评测**（平均长度、Distinct-1/2、格式合规率）
3. **安全性评测**（危险回复率、漏建议就医率）
4. **Markdown 格式的评测报告**

评测结果保存在：

```bash
cat /root/minimind/applications/medical_qa/data/eval_results.json
```

---

## 第十步：生成实验报告

```bash
cd /root/minimind/applications/medical_qa
```

```bash
python run_experiment.py --skip_train --skip_eval
```

这会读取之前的评测结果，生成实验报告：

```bash
cat results/report.md
```

---

## 第十一步：启动 Streamlit Demo（可选）

如果你想体验交互式 Demo 界面：

```bash
cd /root/minimind
```

```bash
streamlit run applications/medical_qa/demo/medical_chat.py \
    --server.port 6006 \
    --server.address 0.0.0.0
```

终端会显示：

```
Local URL: http://localhost:6006
Network URL: http://0.0.0.0:6006
```

### 如何在浏览器中访问？

AutoDL 的实例不能直接访问，需要做端口映射。有三种方法：

#### 方法 1：AutoDL 自带端口映射（最简单）

1. 回到 AutoDL 网页控制台
2. 找到你的实例 -> 点击 **快捷工具**
3. 点击 **自定义服务**
4. 端口号填 `6006`
5. 点 **生成访问链接**
6. 复制链接，在浏览器打开即可看到 Demo

#### 方法 2：JupyterLab 代理地址

直接在浏览器地址栏输入：

```
https://你的JupyterLab地址/proxy/6006/
```

例如：`https://connect.autodl.com/proxy/6006/`

#### 方法 3：SSH 隧道

在你**本地电脑**的终端（不是 AutoDL 的终端）执行：

```bash
ssh -CNg -L 6006:127.0.0.1:6006 root@connect.xxx.autodl.com -p 12345
```

（把 `connect.xxx.autodl.com` 和 `12345` 替换成你实例的实际 SSH 地址和端口）

然后浏览器打开 `http://localhost:6006`

---

## 第十二步：下载结果到本地

### 打包

```bash
cd /root/minimind

tar czf medical_qa_results.tar.gz \
    applications/medical_qa/results/ \
    applications/medical_qa/data/eval_results.json \
    out/lora/lora_medical_512.pth
```

### 下载方式

#### 方式 1：JupyterLab 下载

在 JupyterLab 左侧文件浏览器中，找到 `/root/minimind/medical_qa_results.tar.gz`，右键 -> **Download**。

#### 方式 2：SCP 命令

在你**本地电脑**的终端执行：

```bash
scp -P 12345 root@connect.xxx.autodl.com:/root/minimind/medical_qa_results.tar.gz ./
```

（把 SSH 地址和端口替换成你实例的实际值）

---

## 第十三步：关机（停止计费）

**重要**：不用了务必关机，否则持续扣费！

1. 回到 AutoDL 控制台
2. 找到你的实例
3. 点击 **关机**

注意：
- **关机** = 停止计费，数据保留（下次开机还在）
- **销毁** = 停止计费，数据删除（不可恢复）

---

## 一键脚本（省去手动执行）

如果你不想一条一条复制，可以把以下内容保存为 `setup_and_run.sh`，一次性执行：

```bash
#!/bin/bash
# AutoDL 一键运行方案5
# 用法：bash setup_and_run.sh

set -e  # 遇到错误立即停止

echo "=============================="
echo "  方案5：医疗问答域应用 - AutoDL 一键运行"
echo "=============================="

cd /root/minimind

echo ""
echo "[1/6] 安装依赖..."
pip install -r requirements.txt -q
pip install streamlit -q

echo ""
echo "[2/6] 验证 GPU..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "[3/6] 生成数据..."
cd applications/medical_qa
python data/generate_sample_data.py
python data/prepare_data.py
cd /root/minimind

echo ""
echo "[4/6] 检查基础权重..."
if [ ! -f "out/full_sft_512.pth" ]; then
    echo "基础权重不存在，尝试从 HuggingFace 下载..."
    pip install huggingface_hub -q
    HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='jingyaogong/MiniMind', filename='full_sft_512.pth', local_dir='out')
print('下载完成')
" || echo "下载失败，请手动获取权重，参考 autodl_guide.md 第七步"
fi

echo ""
echo "[5/6] LoRA 微调..."
python -m trainer.train_lora \
    --lora_name lora_medical \
    --data_path applications/medical_qa/data/train.jsonl \
    --epochs 5 --batch_size 16 --learning_rate 1e-4 \
    --hidden_size 512 --num_hidden_layers 8 --max_seq_len 512 \
    --from_weight full_sft

echo ""
echo "[6/6] 运行评测..."
python applications/medical_qa/eval/medical_eval.py \
    --eval_data applications/medical_qa/data/eval_qa.json \
    --tokenizer_path model --save_dir out \
    --hidden_size 512 --num_hidden_layers 8 \
    --lora_weight lora_medical --device cuda

echo ""
echo "=============================="
echo "  全部完成！"
echo "  评测结果：applications/medical_qa/data/eval_results.json"
echo "  LoRA 权重：out/lora/lora_medical_512.pth"
echo "  启动 Demo：streamlit run applications/medical_qa/demo/medical_chat.py --server.port 6006 --server.address 0.0.0.0"
echo "=============================="
```

使用方法：

```bash
cd /root/minimind
bash setup_and_run.sh
```

---

## 常见问题

### Q: CUDA out of memory

减小 batch_size：

```bash
python -m trainer.train_lora --lora_name lora_medical \
    --data_path applications/medical_qa/data/train.jsonl \
    --epochs 5 --batch_size 4 --learning_rate 1e-4 \
    --hidden_size 512 --num_hidden_layers 8 --from_weight full_sft
```

### Q: ModuleNotFoundError: No module named 'xxx'

```bash
pip install xxx
```

### Q: HuggingFace 下载太慢

用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后重新执行下载命令。

### Q: 训练时 loss 不下降

正常现象。100 条数据 + 5 epochs 可能不够，尝试：
- 增加 epochs：`--epochs 20`
- 增加数据量：使用项目自带的 `dataset/lora_medical.jsonl`（如果有的话）
- 调整学习率：`--learning_rate 5e-5`

### Q: Demo 界面打不开

检查端口映射是否正确。AutoDL 控制台 -> 快捷工具 -> 自定义服务 -> 端口填 6006 -> 生成链接。

### Q: 想重新跑一遍

```bash
# 删除之前的结果
rm -rf /root/minimind/out/lora/lora_medical_512.pth
rm -rf /root/minimind/applications/medical_qa/results/
rm -rf /root/minimind/applications/medical_qa/data/eval_results.json

# 重新运行
cd /root/minimind
bash setup_and_run.sh
```

---

## 费用总结

| 步骤 | 预计时长 | 费用 (RTX 3090) |
|------|---------|----------------|
| 安装依赖 | 5 分钟 | ~0.1 元 |
| 生成数据 | 1 分钟 | ~0 元 |
| LoRA 训练 | 5-10 分钟 | ~0.2 元 |
| 评测 | 10-20 分钟 | ~0.4 元 |
| Demo 体验 | 30 分钟 | ~0.6 元 |
| **合计** | **约 1 小时** | **约 1.5 元** |

**记住用完关机！**
