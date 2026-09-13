from typing import TypedDict

from pydantic import BaseModel, Field


class QuestionSet(BaseModel):
    """Questions the agent wants the user to answer before writing."""

    questions: list[str] = Field(
        min_length=3,
        max_length=7,
        description="Short, concrete questions about format, audience, tone, depth and length.",
    )


class PodcastState(TypedDict, total=False):
    sources: list[str]
    initial_prompt: str
    source_text: str
    questions: list[str]
    preferences: dict[str, str]
    script: str
