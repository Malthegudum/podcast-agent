# Podcast Agent

A small learning project for understanding agents with **LangChain + LangGraph**.

The first version does three things:

1. Takes one or more sources plus an optional initial instruction.
2. Reads the sources and generates a small set of clarifying questions.
3. Pauses for the user's answers, then writes a complete podcast script.

The important learning idea is that the model does not control everything. The workflow is explicit:

```text
START
  |
  v
ingest_sources
  |
  v
generate_questions   <- LLM
  |
  v
ask_user             <- human-in-the-loop interrupt
  |
  v
write_script         <- LLM
  |
  v
END
```

## Why LangGraph?

LangChain gives us the model interface and structured output. LangGraph gives us explicit state and orchestration. The `ask_user` node uses an interrupt, so graph execution pauses until the user answers and then resumes from the saved state.

That makes the code useful for learning agents because you can inspect exactly:

- what state exists,
- which step is deterministic,
- which step calls an LLM,
- where a human enters the loop.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
```

Install the project:

```bash
pip install -e .
```

Create your local environment file:

```bash
cp .env.example .env
```

Then put your OpenAI API key in `.env`.

## Run it

With a webpage:

```bash
podcast-agent "https://example.com/article" \
  --prompt "Explain this for economics students in Danish"
```

With local files:

```bash
podcast-agent notes.md paper.pdf \
  --prompt "Make an accessible 15 minute episode"
```

The script is saved to `output/podcast.md` by default.

## Files worth reading first

Start here:

1. `src/podcast_agent/schemas.py` — the state that moves through the graph.
2. `src/podcast_agent/graph.py` — the actual agent workflow.
3. `src/podcast_agent/cli.py` — how the terminal handles the LangGraph interrupt.
4. `src/podcast_agent/sources.py` — deliberately simple source ingestion.

## What this version deliberately does NOT do yet

It produces a podcast **script**, not an MP3. That is intentional for version 0.1: the purpose is to make the agent architecture visible before adding text-to-speech.

Good next steps are:

- add a TTS node that creates an audio file,
- split research and writing into separate agents,
- add citations/grounding checks,
- summarize long sources instead of truncating them,
- add LangSmith tracing,
- replace the CLI with a small web UI.
