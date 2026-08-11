"""CLI entry point for quick local agent tests.

Usage:
    python -m factories --query "1 + 2 等于多少？"
"""

import argparse

from dotenv import load_dotenv

from .agent_factory import create_agent


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run native LangChain agent from CLI")
    parser.add_argument(
        "--query",
        default="1 + 2 等于多少？",
        help="Question to ask the agent",
    )
    args = parser.parse_args()

    agent = create_agent()
    response = agent.invoke(
        {"messages": [{"role": "user", "content": args.query}]}
    )
    messages = response.get("messages", []) if isinstance(response, dict) else []
    if messages:
        print(messages[-1].content)
    else:
        print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
