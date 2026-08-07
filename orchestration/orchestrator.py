"""基于 LangGraph 的心理健康辅导 Agent 编排器.

架构概览
--------
从原 MultiAgentOrchestrator（任务执行流水线）重构为
CounselingOrchestrator（多模态感知 → 意图理解 → RAG 检索 → ReAct 推理 → 安全校验 → 回应）.

图拓扑::

    START → perceive → understand → retrieve ─┐
                              │                │
                              └─(skip RAG)─────┤
                                               ▼
                                            reason ──(max_iter→respond)──┐
                                              │                          │
                                              ▼                          │
                                            safety                       │
                                           ╱      ╲                      │
                                     pass          fail                  │
                                       │             │                   │
                                       ▼             ▼                   │
                                    respond       reason (rewrite)       │
                                       │                                 │
                                       ▼                                 │
                                      END  ◄─────────────────────────────┘

节点职责:
    perceive   — 封装 Block 1 多模态 API, 产出 emotion_features
    understand — 纯 LLM 推理: 意图识别 + 情绪分类
    retrieve   — 封装 Block 2 RAG 检索 API (大部分情况必须调用)
    reason     — ReAct 式 LLM 决策, 从预定义辅导工具集中选择
    safety     — 规则 + LLM 安全检查 (禁止自杀建议/药物推荐/诊断断言)
    respond    — 输出最终回应 → END

可复用遗产:
    - _agent_call (调用 LLM 的封装)
    - _append_react (ReAct 审计追踪)
    - MultiAgentOrchestrator 骨架 (__init__ + _build_graph + run)
    - StateGraph 模式 / 条件边路由签名
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger("agent_service.orchestrator")

from factories import create_agent
from services.multimodal_client import MultimodalClient, EmotionFeatures
from services.rag_client import RAGClient, RAGContext
from tools.counseling_tools import (
    COUNSELING_TOOLS,
    execute_counseling_tool,
    get_tools_schema_text,
)

from .state import CounselingState, default_state

# 可选: 加载 .env (不强制依赖 python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompts
# ═══════════════════════════════════════════════════════════════════════════════

UNDERSTAND_SYSTEM = """\
你是心理健康辅导系统的意图识别与情绪分类模块。

根据用户消息和多模态情绪特征, 完成两项任务:

1. **意图分类** (五选一):
   - seeking_emotional_support: 用户寻求情感支持、倾诉、需要被理解
   - asking_knowledge: 用户询问心理健康知识、技巧、概念
   - crisis_help: 用户表达自伤/自杀念头或严重危机
   - casual_chat: 纯寒暄、打招呼、感谢、告别
   - unclear: 意图不明, 需要追问

2. **情绪标签** (选最匹配的 1-2 个):
   anxiety, depression, anger, sadness, fear, guilt, shame,
   loneliness, stress, grief, neutral, positive, mixed

输出格式 (严格遵守):
```
intent: <意图标签>
emotion: <情绪标签1>, <情绪标签2>
brief: <一句话概括用户当前状态>
```"""

REASON_SYSTEM_TEMPLATE = """\
你是心理健康辅导 AI 助手。你的任务是综合分析以下信息, 选择最合适的辅导策略.

## 你的原则
- 安全第一: 永远不给出药物推荐、诊断断言或自杀建议
- 共情优先: 在给出任何建议前, 先承认用户情绪是合理的
- 循证实践: 优先使用 CBT、正念、动机访谈等有证据支持的方法
- 承认局限: 你不是治疗师, 适时建议寻求专业帮助

## 可用辅导工具
{tools_schema}

## 当前上下文
- 用户消息: {user_query}
- 对话历史: {history_summary}
- 多模态情绪特征: {emotion_summary}
- 情绪标签: {emotion_label}
- 用户意图: {user_intent}
- RAG 检索知识: {rag_context}

## 任务
1. **Thought**: 分析用户当前需要什么类型的帮助
2. **Action**: 从上述工具中选择最合适的一个, 给出工具参数
3. 输出格式:
```
Thought: <你的推理分析>
Action: <工具名>
Params: <JSON 格式的工具参数>
```"""

SAFETY_SYSTEM = """\
你是心理健康辅导系统的安全审核模块。

检查以下草稿回应是否包含不安全的內容:

## 绝对禁止 (任一命中 → unsafe)
1. 药物推荐: 推荐/暗示任何药物 (包括中药、保健品)
2. 诊断断言: "你有XX症"、"你得了XX病"
3. 自杀方法描述: 任何方法细节
4. 替代专业治疗: "你不用看医生"、"心理咨询没用"
5. 保证疗效: "这样做一定有效"、"保证能好"

## 需要关注 (标记但可能仍 safe)
- 是否过度共情变成鼓励无助感
- 是否忽略了用户的危机信号
- 是否给出了超出 AI 能力范围的承诺

## 输出格式
```
verdict: safe | unsafe
issues: <问题列表, 逗号分隔; 若无问题则写 "none">
suggestion: <如果 unsafe, 如何修改; 否则写 "none">
```"""


# ═══════════════════════════════════════════════════════════════════════════════
# CounselingOrchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class CounselingOrchestrator:
    """多模态心理健康辅导 Agent 编排器.

    Usage::

        orch = CounselingOrchestrator()
        result = orch.run(
            user_query="我最近总是睡不着, 一闭眼就想很多事",
            session_id="user-001",
            conversation_history=[...],
        )
        print(result["final_answer"])
    """

    def __init__(
        self,
        multimodal_base_url: str | None = None,
        rag_base_url: str | None = None,
    ) -> None:
        """初始化编排器.

        Args:
            multimodal_base_url: Block 1 多模态 API 基地址.
                默认读取环境变量 MULTIMODAL_BASE_URL.
            rag_base_url: Block 2 RAG API 基地址.
                默认读取环境变量 RAG_BASE_URL.
        """
        # -- 外部服务客户端 -------------------------------------------------
        self.multimodal = MultimodalClient(base_url=multimodal_base_url)
        self.rag = RAGClient(base_url=rag_base_url)

        # -- LLM Agent (用于纯推理节点: understand / reason / safety) -------
        self.understand_agent = create_agent(
            system_message=UNDERSTAND_SYSTEM,
            enable_tools=False,
        )
        self.reason_agent = create_agent(
            system_message="你是心理健康辅导 AI 助手.",
            enable_tools=False,
        )
        self.safety_agent = create_agent(
            system_message=SAFETY_SYSTEM,
            enable_tools=False,
        )

        # -- 编译图 --------------------------------------------------------
        self.graph = self._build_graph()

    # ── 1. Public API ──────────────────────────────────────────────────────

    def run(
        self,
        user_query: str,
        session_id: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_iterations: int = 8,
    ) -> Dict[str, Any]:
        """执行一轮心理健康辅导对话.

        Args:
            user_query: 用户输入文本.
            session_id: 会话 ID (多轮对话追踪).
            conversation_history: 之前的对话历史.
            image_path: 可选的面部图像文件路径 (Block 1).
            audio_path: 可选的语音文件路径 (Block 1).
            max_iterations: reason 节点 ReAct 内部循环上限.

        Returns:
            包含 final_answer, react_trace 等字段的完整状态 dict.
        """
        state = default_state(
            user_query=user_query,
            session_id=session_id,
            max_iterations=max_iterations,
        )
        if conversation_history:
            state["conversation_history"] = conversation_history

        # 将文件路径注入 runtime context（LangGraph state 不持文件路径）
        state["runtime_context"] = {  # type: ignore[typeddict-unknown-key]
            "image_path": image_path,
            "audio_path": audio_path,
        }

        return self.graph.invoke(state)

    # ── 2. Graph assembly ──────────────────────────────────────────────────

    def _build_graph(self):
        """构建 LangGraph StateGraph.

        拓扑:
            START → perceive → understand → retrieve ─┐
                                        │               │
                                        └─(skip)────────┤
                                                        ▼
                                                      reason
                                                        │
                                                        ▼
                                                      safety
                                                     ╱      ╲
                                               pass          fail
                                                 │             │
                                                 ▼             ▼
                                              respond       reason
                                                 │
                                                 ▼
                                                END
        """
        graph = StateGraph(CounselingState)

        # -- 注册节点 -------------------------------------------------------
        graph.add_node("perceive",   self._perceive_node)
        graph.add_node("understand", self._understand_node)
        graph.add_node("retrieve",   self._retrieve_node)
        graph.add_node("reason",     self._reason_node)
        graph.add_node("safety",     self._safety_node)
        graph.add_node("respond",    self._respond_node)

        # -- 固定边 ---------------------------------------------------------
        graph.add_edge(START, "perceive")
        graph.add_edge("perceive", "understand")

        # -- understand → retrieve 或 skip → reason ------------------------
        graph.add_conditional_edges(
            "understand",
            self._retrieve_router,
            {
                "retrieve": "retrieve",
                "skip": "reason",
            },
        )
        graph.add_edge("retrieve", "reason")

        # -- reason → safety (正常) 或 respond (max_iterations 兜底) ---------
        graph.add_conditional_edges(
            "reason",
            self._reason_router,
            {
                "safety": "safety",
                "respond": "respond",
            },
        )

        # -- safety → respond (pass) 或 reason (fail → 重写) ----------------
        graph.add_conditional_edges(
            "safety",
            self._safety_router,
            {
                "respond": "respond",
                "rewrite": "reason",
            },
        )

        # -- respond → END --------------------------------------------------
        graph.add_edge("respond", END)

        return graph.compile()

    # ── 3. Node implementations ────────────────────────────────────────────

    # 3a. perceive — 调用 Block 1 多模态 API ────────────────────────────────

    def _perceive_node(self, state: CounselingState) -> CounselingState:
        """Node 1 — 感知: 调用 Block 1 多模态特征提取 API.

        从 runtime_context 中取文件路径, 调用 MultimodalClient.analyze(),
        将合并后的情绪特征写入 state.
        """
        t0 = time.time()
        log = list(state.get("execution_log", []))
        rt = state.get("runtime_context", {})  # type: ignore[typeddict-unknown-key]

        image_path = rt.get("image_path") if isinstance(rt, dict) else None
        audio_path = rt.get("audio_path") if isinstance(rt, dict) else None
        user_query = state["user_query"]

        # 调用 Block 1
        from pathlib import Path
        features = self.multimodal.analyze(
            text=user_query,
            image_path=Path(image_path) if image_path else None,
            audio_path=Path(audio_path) if audio_path else None,
        )

        log.append(f"perceive modalities={features.available_modalities}")
        if features.error_modalities:
            log.append(f"perceive errors={features.error_modalities}")

        react_trace = self._append_react(
            state,
            thought="调用 Block 1 多模态 API 采集用户情绪特征.",
            action="perceive_multimodal_analysis",
            observation=features.summary(),
        )

        logger.info("perceive elapsed=%.2fs modalities=%s", time.time() - t0, features.available_modalities)

        return {
            "emotion_features": features.to_dict(),
            "status": "understanding",
            "execution_log": log,
            "react_trace": react_trace,
        }

    # 3b. understand — LLM 意图识别 + 情绪分类 ──────────────────────────────

    def _understand_node(self, state: CounselingState) -> CounselingState:
        """Node 2 — 理解: 纯 LLM 推理, 识别用户意图和情绪类别.

        不调用外部服务——意图识别和情绪分类是 LLM 擅长的高层次推理任务.
        """
        t0 = time.time()
        log = list(state.get("execution_log", []))
        features = state.get("emotion_features", {})

        # 构建 emotion summary
        ef = EmotionFeatures(**features) if features else EmotionFeatures()
        emotion_summary = ef.summary()

        prompt = (
            f"用户消息: {state['user_query']}\n"
            f"多模态情绪特征: {emotion_summary}\n"
            f"对话历史轮数: {len(state.get('conversation_history', []))}\n"
            "\n请输出意图标签、情绪标签和简短概括."
        )

        try:
            raw = self._agent_call(self.understand_agent, prompt)
            parsed = self._parse_understand_output(raw)

            log.append(
                f"understand intent={parsed['intent']} "
                f"emotion={parsed['emotion']}"
            )

            react_trace = self._append_react(
                state,
                thought=f"分析用户意图和情绪状态: {parsed['brief']}",
                action="understand_intent_classify",
                observation=(
                    f"intent={parsed['intent']}, "
                    f"emotion={parsed['emotion']}"
                ),
            )

            logger.info("understand elapsed=%.2fs intent=%s emotion=%s", time.time() - t0, parsed["intent"], parsed["emotion"])

            return {
                "user_intent": parsed["intent"],
                "emotion_label": parsed["emotion"],
                "status": "retrieving",
                "execution_log": log,
                "react_trace": react_trace,
            }
        except Exception as exc:
            # 降级: 默认意图 + 中性情绪
            log.append(f"understand_error={exc}")
            react_trace = self._append_react(
                state,
                thought="意图识别失败, 使用默认分类.",
                action="understand_fallback",
                observation=str(exc),
            )
            logger.warning("understand elapsed=%.2fs error=%s", time.time() - t0, exc)
            return {
                "user_intent": "unclear",
                "emotion_label": "neutral",
                "status": "retrieving",
                "execution_log": log,
                "react_trace": react_trace,
            }

    # 3c. retrieve — 调用 Block 2 RAG 检索 API ──────────────────────────────

    def _retrieve_node(self, state: CounselingState) -> CounselingState:
        """Node 3 — 检索: 调用 Block 2 RAG 知识库检索.

        用 user_query + emotion_label 作为组合查询, 检索相关文献/案例.
        """
        t0 = time.time()
        log = list(state.get("execution_log", []))

        # 组合查询
        query_parts = [state["user_query"]]
        emotion = state.get("emotion_label", "")
        if emotion and emotion != "neutral":
            query_parts.append(f"情绪类型: {emotion}")
        combined_query = "; ".join(query_parts)

        rag_ctx: RAGContext = self.rag.retrieve(
            query=combined_query,
            top_k=5,
        )
        log.append(rag_ctx.summary())

        retrieved_docs = [
            {
                "content": d.content,
                "source": d.source,
                "score": d.relevance_score,
            }
            for d in rag_ctx.documents
        ]

        react_trace = self._append_react(
            state,
            thought="根据用户问题和情绪标签检索相关知识.",
            action="retrieve_knowledge",
            observation=(
                f"retrieved {len(rag_ctx.documents)} docs, "
                f"query='{combined_query[:80]}...'"
            ),
        )

        logger.info("retrieve elapsed=%.2fs docs=%d", time.time() - t0, len(rag_ctx.documents))

        return {
            "retrieved_docs": retrieved_docs,
            "rag_context": rag_ctx.formatted_text,
            "status": "reasoning",
            "execution_log": log,
            "react_trace": react_trace,
        }

    # 3d. reason — ReAct 式 LLM 决策 ─────────────────────────────────────────

    def _reason_node(self, state: CounselingState) -> CounselingState:
        """Node 4 — 推理: 综合所有信息, ReAct 式选择辅导工具并生成回应.

        这是唯一允许 LLM 进行"工具选择"的节点, 但工具集限定为
        COUNSELING_TOOLS 中的高层策略函数, 不涉及底层 API 调度.
        """
        t0 = time.time()
        log = list(state.get("execution_log", []))
        iteration = int(state.get("iteration", 0))
        max_iterations = int(state.get("max_iterations", 8))

        # 防止无限重写 (safety fail → rewrite 循环)
        if iteration >= max_iterations:
            log.append("reason_max_iterations")
            logger.info("reason elapsed=%.2fs action=fallback_max_iterations", time.time() - t0)
            # 使用兜底安全回应——直接跳到 responding, 跳过 safety
            # (兜底信息是硬编码的安全文本, 无需再校验)
            return {
                "draft_response": (
                    "感谢你的分享. 作为 AI 助手, 我可能暂时无法完全理解你的情况. "
                    "如果你正在经历困难, 联系信任的朋友、家人或专业心理咨询师 "
                    "会是更好的选择. 我随时在这里倾听你."
                ),
                "safety_passed": True,   # 兜底文本保证安全
                "status": "responding",   # 跳过 safety 节点
                "execution_log": log,
                "react_trace": self._append_react(
                    state,
                    thought="达到最大迭代次数, 使用兜底安全回应.",
                    action="reason_fallback_max_iterations",
                    observation="使用预设安全回应, 跳过安全检查",
                ),
            }

        # -- 构建 history summary ------------------------------------------
        history = state.get("conversation_history", [])
        if history:
            recent = history[-6:]  # 最近 3 轮
            history_summary = "\n".join(
                f"[{h['role']}]: {h['content'][:200]}" for h in recent
            )
        else:
            history_summary = "(首轮对话)"

        # -- emotion summary -----------------------------------------------
        features = state.get("emotion_features", {})
        ef = EmotionFeatures(**features) if features else EmotionFeatures()

        # -- 构建 prompt ---------------------------------------------------
        prompt = REASON_SYSTEM_TEMPLATE.format(
            tools_schema=get_tools_schema_text(),
            user_query=state["user_query"],
            history_summary=history_summary,
            emotion_summary=ef.summary(),
            emotion_label=state.get("emotion_label", "unknown"),
            user_intent=state.get("user_intent", "unclear"),
            rag_context=state.get("rag_context", "(未检索)"),
        )

        try:
            raw = self._agent_call(self.reason_agent, prompt)
            thought, tool_name, tool_params = self._parse_reason_output(raw)

            # 执行选中的辅导工具
            tool_result = execute_counseling_tool(tool_name, tool_params)

            log.append(
                f"reason thought='{thought[:80]}...' "
                f"action={tool_name}"
            )

            react_trace = self._append_react(
                state,
                thought=thought,
                action=f"{tool_name}({tool_params})",
                observation=tool_result["content"][:200],
            )

            # 将工具产出的半成品作为 draft_response
            # (后续可扩展: 再调一次 LLM 将 tool_result 润色为自然对话)
            draft = self._render_draft_response(
                tool_name=tool_name,
                tool_content=tool_result["content"],
                user_query=state["user_query"],
                emotion_label=state.get("emotion_label", ""),
            )

            logger.info("reason elapsed=%.2fs action=%s iteration=%d", time.time() - t0, tool_name, iteration + 1)

            return {
                "agent_thought": thought,
                "agent_action": tool_name,
                "action_params": tool_params,
                "draft_response": draft,
                "iteration": iteration + 1,
                "status": "validating",
                "execution_log": log,
                "react_trace": react_trace,
            }
        except Exception as exc:
            log.append(f"reason_error={exc}")
            logger.warning("reason elapsed=%.2fs error=%s", time.time() - t0, exc)
            react_trace = self._append_react(
                state,
                thought="推理节点出错, 使用兜底回应.",
                action="reason_handle_error",
                observation=str(exc),
            )
            return {
                "draft_response": (
                    "我理解你现在可能有复杂的感受. 虽然我暂时无法给出最合适的回应, "
                    "但我在这里倾听你. 你愿意再多说一些吗?"
                ),
                "iteration": iteration + 1,
                "status": "validating",
                "execution_log": log,
                "react_trace": react_trace,
            }

    # 3e. safety — 安全门禁 ──────────────────────────────────────────────────

    def _safety_node(self, state: CounselingState) -> CounselingState:
        """Node 5 — 安全校验: 规则 + LLM 双重检查草稿回应.

        硬性规则 (确定性的, 不依赖 LLM):
            - 禁止药物名称列表匹配
            - 禁止诊断关键词匹配

        软性规则 (LLM):
            - 检测是否有过度承诺
            - 检测是否忽略了危机信号
        """
        t0 = time.time()
        log = list(state.get("execution_log", []))
        draft = state.get("draft_response", "")

        # -- 第一层: 确定性规则检查 -----------------------------------------
        rule_issues = self._rule_based_safety_check(draft)

        # -- 第二层: LLM 安全检查 -------------------------------------------
        llm_issues: List[str] = []
        try:
            safety_prompt = f"请审核以下草稿回应:\n\n{draft}"
            raw = self._agent_call(self.safety_agent, safety_prompt)
            verdict, llm_issues, suggestion = self._parse_safety_output(raw)
            safety_passed = (verdict == "safe") and (len(rule_issues) == 0)
        except Exception as exc:
            log.append(f"safety_llm_error={exc}")
            # LLM 检查失败时, 如果规则检查通过则放行
            safety_passed = len(rule_issues) == 0
            llm_issues = []
            suggestion = ""

        all_issues = rule_issues + llm_issues
        log.append(
            f"safety passed={safety_passed} "
            f"rule_issues={rule_issues} llm_issues={llm_issues}"
        )

        logger.info("safety elapsed=%.2fs passed=%s issues=%d", time.time() - t0, safety_passed, len(all_issues))

        react_trace = self._append_react(
            state,
            thought=(
                "检查草稿回应是否符合安全规范: "
                + ("通过" if safety_passed else f"发现问题: {all_issues}")
            ),
            action="safety_validate_response",
            observation=f"safety_passed={safety_passed}, issues={all_issues}",
        )

        return {
            "safety_issues": all_issues,
            "safety_passed": safety_passed,
            "status": "responding" if safety_passed else "reasoning",
            "execution_log": log,
            "react_trace": react_trace,
        }

    # 3f. respond — 输出最终回应 ─────────────────────────────────────────────

    def _respond_node(self, state: CounselingState) -> CounselingState:
        """Node 6 — 回应: 将最终回应写入 final_answer, 结束本轮.

        多轮对话由外部调用者管理——调用者从 final_answer 取结果,
        下一次调用 run() 时传入 conversation_history 即可延续对话.
        """
        t0 = time.time()
        log = list(state.get("execution_log", []))

        draft = state.get("draft_response", "")
        safety_issues = state.get("safety_issues", [])

        # 如果还有未解决的安全问题, 在输出前做最后的 strip
        final = draft
        if safety_issues and not state.get("safety_passed", False):
            final = (
                "感谢你的分享. 作为 AI 助手, 我建议你就此咨询专业的心理健康人士. "
                "他们能给你更个性化和安全的指导. 我在这里继续倾听你."
            )
            log.append("respond_fallback_due_to_safety")

        # 更新对话历史
        history = list(state.get("conversation_history", []))
        history.append({"role": "user", "content": state["user_query"]})
        history.append({"role": "assistant", "content": final})

        log.append("respond_complete")
        logger.info("respond elapsed=%.2fs len=%d", time.time() - t0, len(final))
        react_trace = self._append_react(
            state,
            thought="输出经过安全校验的最终回应.",
            action="respond_to_user",
            observation=(final[:150] + "...") if len(final) > 150 else final,
        )

        return {
            "final_answer": final,
            "conversation_history": history,
            "status": "completed",
            "execution_log": log,
            "react_trace": react_trace,
        }

    # ── 4. Conditional routers ──────────────────────────────────────────────

    def _retrieve_router(self, state: CounselingState) -> str:
        """understand → retrieve 或 skip → reason.

        跳过 RAG 的条件:
            - 用户意图为 casual_chat (纯寒暄, 不需要检索)
        """
        intent = state.get("user_intent", "")
        if intent in ("casual_chat",):
            return "skip"
        return "retrieve"

    def _reason_router(self, state: CounselingState) -> str:
        """reason → safety (正常) 或 respond (达到 max_iterations 兜底).

        reason 节点在达到最大迭代次数时设置 status="responding",
        此时跳过安全校验直接输出.
        """
        if state.get("status") == "responding":
            return "respond"
        return "safety"

    def _safety_router(self, state: CounselingState) -> str:
        """safety → respond (通过) 或 reason (不通过 → 重写).

        重写次数由 reason 节点内部的 iteration 计数器控制,
        超过 max_iterations 后 reason 会使用兜底安全回应.
        """
        if state.get("safety_passed", False):
            return "respond"
        return "rewrite"

    # ── 5. Internal helpers ─────────────────────────────────────────────────

    # 5a. LLM call wrapper ──────────────────────────────────────────────────

    def _agent_call(self, agent: Any, prompt: str) -> str:
        """调用 LLM Agent, 返回最后一条消息的文本内容.

        与原始 MultiAgentOrchestrator._agent_call 完全一致——可复用.
        """
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages:
            return str(result)
        return str(messages[-1].content)

    # 5b. ReAct audit trail ─────────────────────────────────────────────────

    def _append_react(
        self,
        state: CounselingState,
        *,
        thought: Optional[str] = None,
        action: Optional[str] = None,
        observation: Optional[str] = None,
    ) -> List[str]:
        """向 ReAct 审计轨迹追加条目.

        与原 _append_react 完全一致——框架无关, 直接复用.
        """
        trace = list(state.get("react_trace", []))
        if thought:
            trace.append(f"Thought: {thought}")
        if action:
            trace.append(f"Action: {action}")
        if observation:
            trace.append(f"Observation: {observation}")
        return trace

    # 5c. Output parsers ────────────────────────────────────────────────────

    def _parse_understand_output(self, text: str) -> Dict[str, str]:
        """解析 understand 节点的 LLM 输出."""
        result = {"intent": "unclear", "emotion": "neutral", "brief": ""}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("intent:"):
                result["intent"] = line.split(":", 1)[1].strip()
            elif line.startswith("emotion:"):
                result["emotion"] = line.split(":", 1)[1].strip()
            elif line.startswith("brief:"):
                result["brief"] = line.split(":", 1)[1].strip()
        return result

    def _parse_reason_output(self, text: str) -> tuple:
        """解析 reason 节点的 ReAct 输出 (Thought / Action / Params)."""
        thought = ""
        action = "empathize_and_normalize"  # 默认
        params: Dict[str, Any] = {}

        import re
        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(r"Action:\s*(\S+)", text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            # 确保 action 在已知工具集中
            if action not in COUNSELING_TOOLS:
                action = "empathize_and_normalize"

        # 尝试解析 Params (JSON)
        params_match = re.search(r"Params:\s*(\{.+?\})", text, re.DOTALL | re.IGNORECASE)
        if params_match:
            try:
                import json
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                params = {}

        # 如果没有显式 Params, 从用户消息中推断基础参数
        if not params:
            params = {
                "emotion_label": "distress",
                "user_concern": "(从用户消息推断)",
            }

        return thought, action, params

    def _parse_safety_output(self, text: str) -> tuple:
        """解析 safety 节点的 LLM 输出.

        Returns:
            (verdict, issues_list, suggestion)
        """
        verdict = "safe"
        issues: List[str] = []
        suggestion = ""

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("verdict:"):
                v = line.split(":", 1)[1].strip().lower()
                # 注意: "unsafe" 包含子串 "safe", 必须精确匹配
                verdict = "safe" if v == "safe" else "unsafe"
            elif line.startswith("issues:"):
                iss = line.split(":", 1)[1].strip()
                if iss and iss.lower() != "none":
                    issues = [i.strip() for i in iss.split(",") if i.strip()]
            elif line.startswith("suggestion:"):
                suggestion = line.split(":", 1)[1].strip()

        return verdict, issues, suggestion

    # 5d. Safety rules ──────────────────────────────────────────────────────

    # 药物名称黑名单（确定性规则, 不依赖 LLM）
    _MEDICATION_BLACKLIST = [
        "舍曲林", "氟西汀", "帕罗西汀", "氟伏沙明", "西酞普兰", "艾司西酞普兰",
        "文拉法辛", "度洛西汀", "米氮平", "曲唑酮", "安非他酮",
        "阿普唑仑", "劳拉西泮", "氯硝西泮", "地西泮", "奥沙西泮",
        "唑吡坦", "佐匹克隆", "艾司佐匹克隆",
        "奥氮平", "喹硫平", "利培酮", "阿立哌唑",
        "碳酸锂", "丙戊酸钠", "拉莫三嗪",
        "赛乐特", "百忧解", "左洛复", "怡诺思", "欣百达", "瑞美隆",
    ]

    _DIAGNOSIS_PATTERNS = [
        "你有抑郁症", "你有焦虑症", "你得了", "你患有", "你的病",
        "你这是典型", "你肯定是", "确诊", "临床诊断",
    ]

    def _rule_based_safety_check(self, text: str) -> List[str]:
        """确定性规则安全检查 (不依赖 LLM)."""
        issues: List[str] = []

        # 检查药物名称
        for med in self._MEDICATION_BLACKLIST:
            if med in text:
                issues.append(f"药物推荐: 提及 '{med}'")

        # 检查诊断断言
        for pattern in self._DIAGNOSIS_PATTERNS:
            if pattern in text:
                issues.append(f"诊断断言: 包含 '{pattern}'")

        # 检查自杀方法关键词
        method_keywords = ["割腕", "跳楼", "上吊", "烧炭", "安眠药 overdose"]
        for kw in method_keywords:
            if kw in text:
                issues.append(f"危险内容: 包含 '{kw}'")

        return issues

    # 5e. Response rendering ─────────────────────────────────────────────────

    def _render_draft_response(
        self,
        tool_name: str,
        tool_content: str,
        user_query: str,
        emotion_label: str,
    ) -> str:
        """将工具产出的半成品渲染为自然对话回应.

        当前为简化版: 直接将工具内容作为草稿, reason 节点不二次调用 LLM
        润色. 后续可扩展为再调一次 LLM 做风格化.
        """
        # 对于 crisis_intervention, 直接使用工具内容 (包含热线信息, 不能改)
        if tool_name == "crisis_intervention":
            return tool_content

        # 其他工具: 用 LLM 将结构化内容转为自然对话
        render_prompt = (
            f"你是心理健康辅导 AI. 请将以下结构化辅导内容转化为自然、"
            f"温暖、共情的对话回应.\n\n"
            f"用户消息: {user_query}\n"
            f"用户情绪: {emotion_label}\n"
            f"辅导内容:\n{tool_content}\n\n"
            f"要求:\n"
            f"- 用口语化的中文\n"
            f"- 先共情, 再给建议\n"
            f"- 不要用编号或列表格式\n"
            f"- 不要做诊断或推荐药物\n"
            f"- 长度控制在 150-300 字"
        )

        try:
            rendered = self._agent_call(self.reason_agent, render_prompt)
            return rendered
        except Exception:
            # 降级: 直接返回工具内容
            return tool_content
