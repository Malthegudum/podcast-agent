from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

from .graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turn sources into a podcast script.")
    parser.add_argument(
        "sources",
        nargs="+",
        help="URLs or local .txt/.md/.pdf files.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional initial instruction for the podcast.",
    )
    parser.add_argument(
        "--output",
        default="output/podcast.md",
        help="Where to save the generated script.",
    )
    return parser.parse_args()


def collect_answers(questions: list[str]) -> dict[str, str]:
    print("\nThe agent has a few questions before writing:\n")
    answers: dict[str, str] = {}
    for index, question in enumerate(questions, start=1):
        answer = input(f"{index}. {question}\n> ").strip()
        answers[question] = answer
        print()
    return answers


def main() -> None:
    load_dotenv()
    args = parse_args()
    graph = build_graph()

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "sources": args.sources,
        "initial_prompt": args.prompt,
    }

    result = graph.invoke(initial_state, config=config)
    interrupt_payload = result["__interrupt__"][0].value
    answers = collect_answers(interrupt_payload["questions"])

    final_state = graph.invoke(Command(resume=answers), config=config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_state["script"], encoding="utf-8")

    print(f"Podcast script written to: {output_path}")


if __name__ == "__main__":
    main()
