"""
MiniMind 医疗问答助手 - Streamlit Demo
基于 MiniMind 26M 参数微型 LLM + LoRA 医疗微调的问答界面

Usage:
    cd applications/medical_qa/demo
    streamlit run medical_chat.py
"""

import sys
import os
import streamlit as st

# ============================================================
# 路径设置：将 minimind-master 根目录加入 sys.path
# ============================================================
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ============================================================
# 危险关键词列表
# ============================================================
DANGEROUS_KEYWORDS = [
    '绝对有效', '肯定能治好', '不需要看医生', '立即停药', '不用去医院',
]

# ============================================================
# 示例问题
# ============================================================
EXAMPLE_QUESTIONS = [
    "头痛的常见原因有哪些？",
    "感冒和流感有什么区别？",
    "高血压患者饮食需要注意什么？",
    "小孩发烧怎么办？",
]


# ============================================================
# 模型加载（带缓存，失败时切换 mock 模式）
# ============================================================
@st.cache_resource(show_spinner="正在加载 MiniMind 模型...")
def load_model(mode: str = "lora"):
    """
    加载 MiniMind 模型。
    - mode="base": 仅加载基础 SFT 模型
    - mode="lora": 加载基础模型 + LoRA 医疗权重

    返回 (model, tokenizer, mock)
    mock=True 表示模型未成功加载，将使用 mock 回复。
    """
    try:
        import torch
        from transformers import AutoTokenizer
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        from model.model_lora import apply_lora, load_lora

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(_PROJECT_ROOT, 'model')
        )

        # 构建模型
        model = MiniMindForCausalLM(
            MiniMindConfig(hidden_size=512, num_hidden_layers=8)
        )

        # 加载基础权重
        base_weight_path = os.path.join(_PROJECT_ROOT, 'out', 'full_sft_512.pth')
        model.load_state_dict(
            torch.load(base_weight_path, map_location='cpu', weights_only=True),
            strict=True,
        )

        # 如果选择 LoRA 模式，额外加载 LoRA 医疗权重
        if mode == "lora":
            lora_path = os.path.join(_PROJECT_ROOT, 'out', 'lora', 'lora_medical_512.pth')
            apply_lora(model)
            load_lora(model, lora_path)

        model.eval()
        return model, tokenizer, False  # mock=False

    except Exception as e:
        st.warning(f"模型加载失败，已进入演示模式（Mock）。错误信息: {e}")
        return None, None, True  # mock=True


def generate_response(prompt: str, model, tokenizer, temperature: float,
                      top_p: float, max_tokens: int, is_mock: bool) -> str:
    """
    根据用户输入生成回答。
    如果 is_mock=True，返回固定的演示回复。
    """
    if is_mock:
        return (
            "【演示模式】模型未加载，请先训练模型并确保权重文件位于 `out/` 目录下。\n\n"
            "训练步骤参考:\n"
            "1. `python trainer/train_full_sft.py`  (基础 SFT)\n"
            "2. `python trainer/train_lora.py --lora_name lora_medical`  (医疗 LoRA 微调)\n\n"
            f"您的问题是: {prompt}"
        )

    import torch

    # 构建对话格式
    conversation = [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=top_p,
            temperature=temperature,
        )

    # 截取生成部分（去掉输入 token）
    response = tokenizer.decode(
        generated_ids[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True,
    )
    return response


def check_dangerous(text: str) -> list:
    """
    检测回答中是否包含危险关键词。
    返回命中的关键词列表。
    """
    return [kw for kw in DANGEROUS_KEYWORDS if kw in text]


# ============================================================
# Streamlit 页面配置
# ============================================================
st.set_page_config(
    page_title="MiniMind 医疗问答助手",
    page_icon="🩺",
    layout="wide",
)

# ============================================================
# 自定义样式
# ============================================================
st.markdown("""
<style>
    /* 主标题 */
    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        font-size: 0.95rem;
        color: #666;
        margin-top: 0;
    }
    /* 模型信息卡片 */
    .info-card {
        background: linear-gradient(135deg, #e8f0fe 0%, #f0f7ff 100%);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .info-card h4 {
        margin: 0 0 8px 0;
        color: #1a73e8;
    }
    /* 安全提示 */
    .safety-box {
        background: #fff8e1;
        border-left: 4px solid #ffb300;
        border-radius: 6px;
        padding: 12px;
        font-size: 0.85rem;
        color: #795548;
    }
    /* 底部免责声明 */
    .disclaimer {
        text-align: center;
        color: #e65100;
        font-weight: 600;
        padding: 12px;
        background: #fff3e0;
        border-radius: 8px;
        margin-top: 16px;
    }
    /* 危险警告 */
    .danger-warning {
        background: #ffebee;
        border: 2px solid #f44336;
        border-radius: 8px;
        padding: 12px;
        color: #c62828;
        font-weight: 600;
    }
    /* 示例问题按钮样式 */
    .example-btn {
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State 初始化
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    # ---- 模型信息卡片 ----
    st.markdown("""
    <div class="info-card">
        <h4>MiniMind 医疗问答模型</h4>
        <p style="margin:2px 0;"><b>模型类型:</b> Causal LM (GPT-like)</p>
        <p style="margin:2px 0;"><b>隐藏维度:</b> 512</p>
        <p style="margin:2px 0;"><b>层数:</b> 8</p>
        <p style="margin:2px 0;"><b>注意力头:</b> 8 (KV heads: 2)</p>
        <p style="margin:2px 0;"><b>词表大小:</b> 6400</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- 模式切换 ----
    st.markdown("---")
    st.subheader("模型模式")
    model_mode = st.radio(
        "选择加载模式",
        options=["lora", "base"],
        format_func=lambda x: "LoRA 医疗微调模型" if x == "lora" else "基础 SFT 模型",
        index=0,
        help="LoRA 模式加载医疗领域微调权重；Base 模式仅使用通用 SFT 权重。",
    )

    # ---- 参数调节 ----
    st.markdown("---")
    st.subheader("生成参数")

    temperature = st.slider(
        "Temperature（温度）",
        min_value=0.01, max_value=1.5, value=0.85, step=0.01,
        help="值越低回答越确定，值越高回答越随机多样。",
    )
    top_p = st.slider(
        "Top-P（核采样）",
        min_value=0.1, max_value=1.0, value=0.85, step=0.01,
        help="保留概率质量前 top_p 的 token 进行采样。",
    )
    max_tokens = st.slider(
        "Max Tokens（最大生成长度）",
        min_value=64, max_value=2048, value=512, step=64,
        help="模型单次回答的最大 token 数。",
    )

    # ---- 安全提示 ----
    st.markdown("---")
    st.markdown("""
    <div class="safety-box">
        <b>安全提示</b><br>
        本系统为 MiniMind 微型模型（26M 参数）的演示应用，<b>不能替代专业医疗诊断</b>。
        如有身体不适，请及时就医。模型输出仅供参考，请勿作为用药或治疗依据。
    </div>
    """, unsafe_allow_html=True)

    # ---- 示例问题 ----
    st.markdown("---")
    st.subheader("示例问题")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_prompt = q
            st.rerun()


# ============================================================
# 主界面
# ============================================================

# 标题
st.markdown('<p class="main-title">MiniMind 医疗问答助手</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">基于 26M 参数微型大模型 + LoRA 医疗领域微调</p>',
    unsafe_allow_html=True,
)

# 指标卡片
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("模型参数量", "~26M")
with col2:
    st.metric("LoRA 秩 (Rank)", 8)
with col3:
    st.metric("生成温度", f"{temperature:.2f}")
with col4:
    st.metric("最大 Token", max_tokens)

st.markdown("---")


# ============================================================
# 聊天历史展示
# ============================================================
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])


# ============================================================
# 处理用户输入（来自侧边栏示例按钮或聊天输入框）
# ============================================================
# 从侧边栏示例按钮获取
if st.session_state.pending_prompt is not None:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
else:
    user_input = st.chat_input("请输入您的医疗健康问题...")

if user_input:
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 加载模型（带缓存，根据模式切换）
    model, tokenizer, is_mock = load_model(model_mode)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在思考中..."):
            answer = generate_response(
                prompt=user_input,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                is_mock=is_mock,
            )

        # 显示回答
        st.markdown(answer)

        # 危险关键词检测
        danger_hits = check_dangerous(answer)
        if danger_hits:
            st.markdown(f"""
            <div class="danger-warning">
                ⚠️ 回答中包含可能误导的表述: <b>{'、'.join(danger_hits)}</b><br>
                请务必咨询专业医生，切勿自行诊断或用药！
            </div>
            """, unsafe_allow_html=True)

    # 保存助手消息
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ============================================================
# 底部免责声明（始终显示）
# ============================================================
st.markdown("""
<div class="disclaimer">
    ⚠️ 本回答仅供参考，不构成医疗建议，请咨询专业医生
</div>
""", unsafe_allow_html=True)
