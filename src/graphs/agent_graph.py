import operator
from typing import Annotated, Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ──────────────────────────────────────────────────────────────────────────────
# State schema
# ──────────────────────────────────────────────────────────────────────────────

TaskStatus = Literal["pending", "in_progress", "completed", "failed"]


class Task(BaseModel):
    id: str
    description: str
    assigned_agent: str
    status: TaskStatus = "pending"
    output: str | None = None


def _replace_tasks(current: list[Task], update: list[Task]) -> list[Task]:
    """Merge task updates by id; append new tasks that don't exist yet."""
    updated_by_id = {t.id: t for t in update}
    merged = [updated_by_id.pop(t.id, t) for t in current]
    merged.extend(updated_by_id.values())
    return merged


class AgentState(TypedDict):
    user_objective: str
    tasks_list: Annotated[list[Task], _replace_tasks]
    global_memory: Annotated[list[str], operator.add]
    current_agent: str


# ──────────────────────────────────────────────────────────────────────────────
# Routing functions
# ──────────────────────────────────────────────────────────────────────────────

# Agents handled by the main task router. "reviewer" is intentionally excluded:
# QA is always triggered by the fixed coder → qa edge, never by the router.
_ROUTABLE: dict[str, str] = {
    "researcher": "research",
    "coder":      "coder",
}


def _task_router(state: AgentState) -> str:
    """Scan tasks in order and return the node name for the first pending,
    routable task. Returns END when every task is done."""
    for task in state["tasks_list"]:
        if task.status == "pending" and task.assigned_agent in _ROUTABLE:
            return _ROUTABLE[task.assigned_agent]
    return END


def _qa_router(state: AgentState) -> str:
    """Called after qa_agent.

    qa_agent signals intent through task status, not memory strings:
      • coder task == 'pending'  →  retry requested  →  route to coder
      • coder task == anything else  →  no retry  →  main task router

    This decouples the router from the exact text format of memory entries.
    """
    retry_requested = any(
        t.assigned_agent == "coder" and t.status == "pending"
        for t in state["tasks_list"]
    )
    return "coder" if retry_requested else _task_router(state)


# ──────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    # Lazy imports break the circular dependency:
    #   agent_graph → orchestrator/specialists → agent_graph (Task, AgentState)
    # By this point Task and AgentState are already defined above, so the
    # partial module returned by Python's import system is complete enough.
    from src.agents.orchestrator import orchestrator_node
    from src.agents.specialists import coder_agent, qa_agent, research_agent

    graph = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("research",     research_agent)
    graph.add_node("coder",        coder_agent)
    graph.add_node("qa",           qa_agent)

    # ── Edges ────────────────────────────────────────────────────────────────

    # Entry point
    graph.add_edge(START, "orchestrator")

    # Orchestrator always feeds the task router first.
    graph.add_conditional_edges(
        "orchestrator",
        _task_router,
        {"research": "research", "coder": "coder", END: END},
    )

    # Research loops back through the task router until no research tasks remain,
    # then advances to the next task type (coder) or ends.
    graph.add_conditional_edges(
        "research",
        _task_router,
        {"research": "research", "coder": "coder", END: END},
    )

    # Coder output always goes to QA — unconditional.
    graph.add_edge("coder", "qa")

    # QA branches: retry coder on failure, advance otherwise.
    graph.add_conditional_edges(
        "qa",
        _qa_router,
        {"coder": "coder", "research": "research", END: END},
    )

    return graph


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

# Lazily compiled — importing this module (or specialists.py) no longer
# triggers the full import chain at module-load time, which breaks the
# mutual dependency: agent_graph ↔ orchestrator/specialists.
_workflow_instance = None


def get_workflow():
    """Return the compiled workflow, building it on the first call."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = _build_graph().compile()
    return _workflow_instance


def run_workflow(objective: str) -> AgentState:
    """Run the full multi-agent workflow for a given objective.

    Blocks until all tasks are completed (or the graph reaches END) and
    returns the final AgentState.

    For streaming intermediate steps use the compiled graph directly:
        for step in workflow.stream(initial_state):
            ...
    """
    initial_state: AgentState = {
        "user_objective": objective,
        "tasks_list":     [],
        "global_memory":  [],
        "current_agent":  "",
    }
    return get_workflow().invoke(initial_state)
