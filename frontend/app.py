"""
情绪识别与关怀助手 - 升级版前端
前端负责人 - 成员4 开发

对接真实 Agent API (成员3 的 Jinzhongzi_RAG 服务)
支持：多轮对话、历史记录、情绪变化曲线、图片上传
"""

import streamlit as st
import requests
import plotly.express as px
import pandas as pd
from datetime import datetime
import json

# ============================================================
# 配置区
# ============================================================
AGENT_API_URL = "http://101.34.68.33:8003/v1/agent/analyze"  # 成员3的接口地址

# 情绪分数映射
SENTIMENT_SCORE_MAP = {
    "positive": 3,
    "neutral": 2,
    "mixed": 1.5,
    "negative": 1,
}

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="心理健康分析助手",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 情绪识别与关怀助手")
st.caption("基于多模态 Agentic RAG 的心理健康分析系统")

# ============================================================
# 初始化会话状态
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []  # 聊天历史
if "mood_history" not in st.session_state:
    st.session_state.mood_history = []  # 情绪分数历史（用于画图）

# ============================================================
# 侧边栏：情绪变化曲线
# ============================================================
with st.sidebar:
    st.subheader("📈 情绪变化曲线")

    if st.session_state.mood_history:
        df = pd.DataFrame(st.session_state.mood_history)
        df["score"] = df["sentiment"].map(SENTIMENT_SCORE_MAP)

        fig = px.line(
            df, x="time", y="score", markers=True,
            title="对话情绪趋势",
            labels={"score": "情绪分数", "time": "时间"},
        )
        fig.update_yaxes(
            range=[0, 4],
            tickvals=[1, 1.5, 2, 3],
            ticktext=["负面", "混合", "中性", "正面"],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("开始对话后，这里会显示情绪变化曲线")

    # 清空对话按钮
    if st.button("🔄 清空对话"):
        st.session_state.messages = []
        st.session_state.mood_history = []
        st.rerun()

# ============================================================
# 后端健康检查
# ============================================================
try:
    health = requests.get(f"{AGENT_API_URL.replace('/v1/agent/analyze', '')}/health", timeout=3)
    if health.status_code == 200:
        deps = health.json().get("dependencies", {})
        st.caption(
            f"🟢 系统在线 | 面部分析: {'🟢' if deps.get('vision_service') == 'online' else '🔴'} | "
            f"知识检索: {'🟢' if deps.get('knowledge_service') == 'online' else '🔴'}"
        )
    else:
        st.caption("🔴 后端服务异常")
except Exception:
    st.caption("🔴 无法连接后端服务")

st.divider()

# ============================================================
# 主界面：聊天记录展示
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 用户消息直接展示
        if msg["role"] == "user":
            st.write(msg["content"])
            if msg.get("image"):
                st.image(msg["image"], width=200, caption="上传的图片")

        # 助手消息：丰富展示
        elif msg["role"] == "assistant":
            data = msg.get("data", {})
            analysis = data.get("analysis", {})

            # 两列布局
            col1, col2 = st.columns([1, 2])

            with col1:
                image_emotion = analysis.get("image_emotion", {})
                if image_emotion:
                    st.metric("面部情绪", image_emotion.get("dominant_emotion", "N/A"))
                    st.metric(
                        "微笑强度(AU12)",
                        f"{image_emotion.get('au12_r_smile_intensity', 0):.1f}",
                    )

                text_sentiment = analysis.get("text_sentiment", "N/A")
                st.metric("文本情绪", text_sentiment)

                decision = data.get("decision", "N/A")
                st.metric("Agent决策", decision)

            with col2:
                st.markdown("**🧠 分析回复：**")
                st.info(data.get("reply", ""))

                if analysis.get("text_keywords"):
                    st.caption(f"关键词：{', '.join(analysis['text_keywords'])}")

                if data.get("advice_source"):
                    st.caption(f"📚 {data['advice_source']}")

                # 展开查看完整 JSON
                with st.expander("🔍 查看完整分析数据"):
                    st.json(data)

# ============================================================
# 底部输入区
# ============================================================
with st.container():
    col_text, col_image = st.columns([3, 1])

    with col_text:
        user_text = st.chat_input("输入你的心情描述...")

    with col_image:
        uploaded_image = st.file_uploader(
            "上传人脸照片（可选）",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

# ============================================================
# 处理用户输入
# ============================================================
if user_text:
    # 1. 添加用户消息到历史
    user_msg = {
        "role": "user",
        "content": user_text,
        "time": datetime.now().strftime("%H:%M:%S"),
    }

    # 保存上传的图片
    if uploaded_image:
        user_msg["image"] = uploaded_image

    st.session_state.messages.append(user_msg)

    # 2. 调用 Agent API
    with st.spinner("正在分析中..."):
        try:
            files = {}
            data = {"text": user_text}

            if uploaded_image:
                files["image"] = (
                    uploaded_image.name,
                    uploaded_image.getvalue(),
                    uploaded_image.type,
                )

            resp = requests.post(AGENT_API_URL, data=data, files=files, timeout=60)

            if resp.status_code == 200:
                result = resp.json()
                assistant_msg = {
                    "role": "assistant",
                    "content": result.get("data", {}).get("reply", ""),
                    "data": result.get("data", {}),
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state.messages.append(assistant_msg)

                # 记录情绪历史
                text_sentiment = (
                    result.get("data", {})
                    .get("analysis", {})
                    .get("text_sentiment", "neutral")
                )
                st.session_state.mood_history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "sentiment": text_sentiment,
                })
            else:
                st.error(f"接口返回错误: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ 无法连接到后端服务，请确认 Agent 服务已启动。")
        except Exception as e:
            st.error(f"请求失败: {str(e)}")

    st.rerun()

# ============================================================
# 底部状态栏
# ============================================================
st.divider()
st.caption(f"Agent API: {AGENT_API_URL}")
