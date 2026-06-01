import uuid
from typing import Any

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.graphs.agent_graph import AgentState, Task

load_dotenv()

# ---------------------------------------------------------------------------
# Available specialist agents the orchestrator can delegate work to.
# Keep this list in sync with the specialists defined in specialists.py.
# ---------------------------------------------------------------------------
AVAILABLE_AGENTS: list[str] = [
    "researcher",
    "analyst",
    "writer",
    "coder",
    "reviewer",
]

_SYSTEM_PROMPT = """\
You are the Orchestrator — the central planning agent of a multi-agent system.

YOUR ROLE
---------
Given a high-level user objective, your sole responsibility is to produce a
precise, ordered plan: a list of discrete tasks that, when executed in
sequence by specialist agents, will fully achieve the objective.

AVAILABLE SPECIALIST AGENTS
----------------------------
{agents}

Agent capabilities:
  • researcher  — gathers facts, searches knowledge bases, retrieves context
  • analyst     — interprets data, identifies patterns, draws conclusions
  • writer      — drafts, edits, and formats text-based deliverables
  • coder       — writes, debugs, and reviews code in any language
  • reviewer    — validates outputs for quality, correctness, and consistency

TASK DECOMPOSITION RULES
-------------------------
1. Every task must map to exactly ONE specialist agent.
2. Tasks must be ordered so that each task's inputs are satisfied by previous
   tasks' outputs.
3. Descriptions must be self-contained — a specialist must understand what to
   do from the description alone, without reading other tasks.
4. Aim for the minimum number of tasks needed. Do not split work that a single
   agent can handle in one step.
5. Do not include tasks for work that is outside the stated objective.

OUTPUT
------
You will return structured data only — no prose outside the defined fields.
Populate `reasoning` with a brief chain-of-thought (2–4 sentences) before
listing tasks; this reasoning is for auditability and is not shown to the user.
"""

_HUMAN_TEMPLATE = "User objective: {objective}"


# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------

class _TaskDraft(BaseModel):
    """Single task as produced by the LLM — id and runtime fields are added
    by the orchestrator node, not by the model."""

    description: str = Field(
        description="Clear, self-contained description of what the agent must do."
    )
    assigned_agent: str = Field(
        description=(
            "One of: researcher, analyst, writer, coder, reviewer. "
            "Must match an available specialist exactly."
        )
    )


class _TaskDecomposition(BaseModel):
    """Wrapper returned by `with_structured_output`."""

    reasoning: str = Field(
        description="Brief chain-of-thought explaining the decomposition strategy."
    )
    tasks: list[_TaskDraft] = Field(
        description="Ordered list of tasks that together achieve the objective.",
        min_length=1,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_llm() -> Any:
    return ChatOpenAI(
        model="anthropic/claude-3.7-sonnet",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=4096,
    ).with_structured_output(_TaskDecomposition)


_llm = _build_llm()


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def orchestrator_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: decomposes `user_objective` into a structured task list."""
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT.format(agents=", ".join(AVAILABLE_AGENTS))),
        HumanMessage(content=_HUMAN_TEMPLATE.format(objective=state["user_objective"])),
    ]

    result: _TaskDecomposition = _llm.invoke(messages)

    tasks = [
        Task(
            id=str(uuid.uuid4()),
            description=draft.description,
            assigned_agent=draft.assigned_agent,
            status="pending",
        )
        for draft in result.tasks
    ]

    return {
        "tasks_list": tasks,
        "global_memory": [f"Orchestrator reasoning: {result.reasoning}"],
        "current_agent": "orchestrator",
    }
