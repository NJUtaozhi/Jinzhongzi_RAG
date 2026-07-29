"""CLI entry for running the counseling orchestrator.

Usage:
    python -m orchestration --query "我最近总是失眠, 很焦虑"
    python -m orchestration --query "如何缓解压力" --session-id user-001
    python -m orchestration --query "我今天心情很好, 谢谢你的帮助" \
        --session-id user-001  # 继续之前的对话
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from . import get_orchestrator


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="多模态心理健康辅导 Agent (CLI 演示模式)"
    )
    parser.add_argument(
        "--query",
        default="我最近总是睡不着, 一闭眼就想很多事, 感觉很焦虑",
        help="用户输入文本",
    )
    parser.add_argument(
        "--session-id",
        default="cli-demo",
        help="会话 ID (用于多轮对话追踪)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="可选: 面部图像文件路径 (Block 1 多模态)",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="可选: 语音文件路径 (Block 1 多模态)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="reason 节点最大 ReAct 循环次数",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互模式: 持续对话直到输入 /quit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出完整的 ReAct 审计轨迹",
    )
    args = parser.parse_args()

    CounselingOrchestrator = get_orchestrator()
    orch = CounselingOrchestrator()

    if args.interactive:
        return _interactive_loop(orch, args)

    # 单轮模式
    result = orch.run(
        user_query=args.query,
        session_id=args.session_id,
        image_path=args.image,
        audio_path=args.audio,
        max_iterations=args.max_iterations,
    )

    _print_result(result, verbose=args.verbose)
    return 0


def _interactive_loop(orch, args) -> int:
    """交互式对话循环."""
    print("=" * 60)
    print("  心理健康辅导 Agent — 交互模式")
    print("  输入 /quit 退出, /trace 查看审计轨迹")
    print("=" * 60)

    conversation_history = []
    turn = 0

    while True:
        try:
            user_input = input(f"\n[你] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            print("再见!")
            break
        if user_input.lower() == "/trace":
            print("(暂无上一轮的审计轨迹)")
            continue

        turn += 1
        result = orch.run(
            user_query=user_input,
            session_id=args.session_id,
            conversation_history=conversation_history,
            image_path=args.image,
            audio_path=args.audio,
            max_iterations=args.max_iterations,
        )

        # 更新对话历史（从 state 中获取完整历史）
        conversation_history = result.get("conversation_history", [])

        print(f"\n[AI] {result.get('final_answer', '(无回应)')}")
        print(f"     [状态: {result.get('status')}, "
              f"意图: {result.get('user_intent')}, "
              f"情绪: {result.get('emotion_label')}]")

        if hasattr(args, 'verbose') and args.verbose:
            _print_trace(result)

    return 0


def _print_result(result: dict, verbose: bool = False) -> None:
    """输出单轮结果."""
    print(result.get("final_answer", "(无最终回答)"))
    if verbose:
        _print_trace(result)


def _print_trace(result: dict) -> None:
    """输出 ReAct 审计轨迹."""
    trace = result.get("react_trace", [])
    if trace:
        print("\n" + "-" * 40)
        print("ReAct 审计轨迹:")
        for entry in trace:
            print(f"  {entry}")
        print("-" * 40)


if __name__ == "__main__":
    raise SystemExit(main())
