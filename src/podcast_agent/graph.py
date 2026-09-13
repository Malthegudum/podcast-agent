from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .schemas import PodcastState, QuestionSet
from .sources import load_sources


QUESTION_SYSTEM_PROMPT = """You are the planning stage of a podcast-making agent.
Your job is NOT to write the podcast yet.
Read the supplied sources and initial user instruction, then ask only the questions
whose answers would materially change the final podcast.

Cover the most useful uncertainties, such as audience, duration, tone, host format,
technical depth, language, use of examples, and whether the episode should challenge
or simply explain the source material.

Do not ask for information already clearly specified by the user.
Return between 3 and 7 concise questions.
"""


SCRIPT_SYSTEM_PROMPT = """You are the writing stage of a podcast-making agent.
Create a complete podcast SCRIPT grounded in the supplied source material and the
user's answers. Do not invent facts that are unsupported by the source context.

The script should sound natural when spoken aloud. Use clear transitions and a strong
opening and ending. Respect the requested audience, tone, depth, language, format and
length. If the source material disagrees internally, make that disagreement explicit.

Return only the finished podcast script in Markdown.
"""


def _model():
    model_name = os.getenv("PODCAST_MODEL", "openai:gpt-5.6-luna")
    return init_chat_model(model_name)


def ingest_sources(state: PodcastState) -> PodcastState:
    return {"source_text": load_sources(state["sources"])}


def generate_questions(state: PodcastState) -> PodcastState:
    model = _model().with_structured_output(QuestionSet)
    result = model.invoke(
        [
            SystemMessage(content=QUESTION_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Initial instruction:\n{state.get('initial_prompt') or '(none)'}\n\n"
                    f"Source material:\n{state['source_text']}"
                )
            ),
        ]
    )
    return {"questions": result.questions}


def ask_user(state: PodcastState) -> PodcastState:
    """Pause the graph and let the CLI/UI collect the user's answers."""
    answers = interrupt({"questions": state["questions"]})
    return {"preferences": answers}


def write_script(state: PodcastState) -> PodcastState:
    preferences = "\n".join(
        f"- {question}: {answer}"
        for question, answer in state["preferences"].items()
    )

    response = _model().invoke(
        [
            SystemMessage(content=SCRIPT_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Initial instruction:\n{state.get('initial_prompt') or '(none)'}\n\n"
                    f"User preferences:\n{preferences}\n\n"
                    f"Source material:\n{state['source_text']}"
                )
            ),
        ]
    )
    return {"script": response.text}


def build_graph():
    builder = StateGraph(PodcastState)
    builder.add_node("ingest_sources", ingest_sources)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("ask_user", ask_user)
    builder.add_node("write_script", write_script)

    builder.add_edge(START, "ingest_sources")
    builder.add_edge("ingest_sources", "generate_questions")
    builder.add_edge("generate_questions", "ask_user")
    builder.add_edge("ask_user", "write_script")
    builder.add_edge("write_script", END)

    return builder.compile(checkpointer=InMemorySaver())
