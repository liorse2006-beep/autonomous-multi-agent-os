import os
import subprocess
import sys
import tempfile
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.graphs.agent_graph import AgentState, Task

load_dotenv()

_MAX_RETRIES = 3          # maximum coder → qa cycles before permanent failure
_EXEC_TIMEOUT = 15        # seconds before subprocess is killed


# ──────────────────────────────────────────────────────────────────────────────
# Sandbox execution
# ──────────────────────────────────────────────────────────────────────────────

def _execute_code(code: str) -> tuple[bool, str]:
    """Write `code` to a temp file and run it in an isolated subprocess.

    Returns (success, output) where `output` is stdout on success or the full
    stderr + stdout error log on failure. The subprocess uses the same Python
    interpreter as the current process so all venv packages are available.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=_EXEC_TIMEOUT,
        )

        if proc.returncode == 0:
            return True, proc.stdout or "(execution succeeded — no stdout)"

        # Combine stderr first (tracebacks live there), then stdout
        error_log = "\n".join(filter(None, [proc.stderr.strip(), proc.stdout.strip()]))
        return False, error_log or f"Process exited with code {proc.returncode} (no output)"

    except subprocess.TimeoutExpired:
        return False, (
            f"TimeoutError: execution exceeded the {_EXEC_TIMEOUT}s limit.\n"
            "The code may contain an infinite loop or blocking call."
        )
    except Exception as exc:
        return False, f"SubprocessError: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Search tool
# ──────────────────────────────────────────────────────────────────────────────

@tool
def mock_web_search(query: str) -> str:
    """Search the web and return relevant information snippets."""
    return (
        f"[Search: '{query}']\n"
        "• Finding 1: Key concepts and foundational background relevant to the query.\n"
        "• Finding 2: Best practices and common implementation patterns in production systems.\n"
        "• Finding 3: Recent developments, popular libraries, and tooling options.\n"
        "• Finding 4: Common pitfalls, anti-patterns, and how to avoid them.\n"
    )


def _get_search_tool() -> Any:
    """Return Tavily if TAVILY_API_KEY is set, otherwise fall back to the mock tool."""
    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
            return TavilySearchResults(max_results=4)
        except ImportError:
            pass
    return mock_web_search


# ──────────────────────────────────────────────────────────────────────────────
# LLM factory
# ──────────────────────────────────────────────────────────────────────────────

def _llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model="anthropic/claude-3.7-sonnet",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=8096,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Structured output schemas
# ──────────────────────────────────────────────────────────────────────────────

class _ResearchOutput(BaseModel):
    insights: list[str] = Field(
        description=(
            "Distinct, concrete, actionable insights extracted from the search "
            "results. Each string is a single self-contained insight."
        )
    )
    summary: str = Field(
        description="One-paragraph synthesis of all findings."
    )


class _CodeOutput(BaseModel):
    code: str = Field(
        description=(
            "Complete, immediately runnable code that implements the task. "
            "Include all imports. No placeholders or TODOs."
        )
    )
    explanation: str = Field(
        description="One-paragraph explanation of the key design choices."
    )


class _QAOutput(BaseModel):
    verdict: Literal["passed", "failed"] = Field(
        description=(
            "'passed' only if the code is correct, complete, and ready to ship. "
            "'failed' if any issue — even minor — was found."
        )
    )
    issues: list[str] = Field(
        description="Specific bugs or quality problems. Empty list when verdict is 'passed'."
    )
    feedback: str = Field(
        description="Overall review summary with concrete improvement suggestions."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_pending_task(tasks: list[Task], agent_name: str) -> Task | None:
    """Return the first pending task assigned to `agent_name`, or None."""
    return next(
        (t for t in tasks if t.assigned_agent == agent_name and t.status == "pending"),
        None,
    )


def _run_tool_loop(
    llm_with_tools: Any,
    messages: list,
    tools_by_name: dict[str, Any],
    max_iterations: int = 5,
) -> list:
    """Drive a ReAct-style tool-calling loop until the model stops issuing tool calls."""
    for _ in range(max_iterations):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            break

        for call in response.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return messages


# ──────────────────────────────────────────────────────────────────────────────
# research_agent
# ──────────────────────────────────────────────────────────────────────────────

_RESEARCH_SYSTEM = """\
You are a Research Specialist in a multi-agent pipeline.

Your job is to gather comprehensive, accurate information using the search tool.
Run multiple targeted searches until you have sufficient coverage to fully address
the task. Prefer specificity over breadth: extract concrete details, not summaries.

When you have gathered enough information, stop calling the tool. Your final
structured output must contain distinct, actionable insights — not generic statements.
"""


def research_agent(state: AgentState) -> dict[str, Any]:
    """Finds the first pending researcher task, gathers information via search,
    and appends concrete insights to global_memory."""
    task = _find_pending_task(state["tasks_list"], "researcher")
    if task is None:
        return {}

    search_tool = _get_search_tool()
    tools = [search_tool]
    tools_by_name: dict[str, Any] = {t.name: t for t in tools}

    base_llm = _llm()
    llm_with_tools = base_llm.bind_tools(tools)

    messages = _run_tool_loop(
        llm_with_tools,
        messages=[
            SystemMessage(content=_RESEARCH_SYSTEM),
            HumanMessage(content=task.description),
        ],
        tools_by_name=tools_by_name,
    )

    # Second pass: extract structured insights from the full conversation history.
    structured_llm = base_llm.with_structured_output(_ResearchOutput)
    result: _ResearchOutput = structured_llm.invoke(
        messages
        + [
            HumanMessage(
                content=(
                    "Based on the research above, extract the key insights and "
                    "write a synthesis in the required structured format."
                )
            )
        ]
    )

    updated_task = task.model_copy(update={"status": "completed", "output": result.summary})

    memory_entries = [f"[Research — {task.id}] {insight}" for insight in result.insights]

    return {
        "tasks_list": [updated_task],
        "global_memory": memory_entries,
        "current_agent": "researcher",
    }


# ──────────────────────────────────────────────────────────────────────────────
# coder_agent
# ──────────────────────────────────────────────────────────────────────────────

_CODER_SYSTEM = """\
You are a Code Specialist in a multi-agent pipeline.

Guidelines:
  • Write clean, modular, production-quality code.
  • Follow SOLID principles and language-specific idioms.
  • Include ALL necessary imports — the code must run as-is.
  • Use the research insights provided to inform implementation decisions.
  • Add comments only where the reasoning is non-obvious.
  • No placeholders, no stub implementations, no TODOs.
"""


def coder_agent(state: AgentState) -> dict[str, Any]:
    """Finds the first pending coder task, writes clean code informed by
    research insights in global_memory, and stores the output on the task.
    On retry cycles the QA error log is injected so the model fixes the
    exact reported problem."""
    task = _find_pending_task(state["tasks_list"], "coder")
    if task is None:
        return {}

    research_context = "\n".join(
        e for e in state["global_memory"] if e.startswith("[Research")
    )
    qa_failures = "\n\n".join(
        e for e in state["global_memory"] if "[QA —" in e and "FAILED" in e
    )

    sections: list[str] = [f"Task:\n{task.description}"]

    if research_context:
        sections.append(f"Research context:\n{research_context}")
    else:
        sections.append("No prior research available — rely on general knowledge.")

    if qa_failures:
        sections.append(
            "PREVIOUS ATTEMPTS FAILED — you MUST fix every issue listed below "
            "before submitting again:\n\n" + qa_failures
        )

    structured_llm = _llm().with_structured_output(_CodeOutput)

    result: _CodeOutput = structured_llm.invoke([
        SystemMessage(content=_CODER_SYSTEM),
        HumanMessage(content="\n\n".join(sections)),
    ])

    updated_task = task.model_copy(update={"status": "completed", "output": result.code})

    return {
        "tasks_list": [updated_task],
        "global_memory": [f"[Code — {task.id}] {result.explanation}"],
        "current_agent": "coder",
    }


# ──────────────────────────────────────────────────────────────────────────────
# qa_agent  (The Critic)
# ──────────────────────────────────────────────────────────────────────────────

_QA_SYSTEM = """\
You are a QA Specialist — The Critic.

The code has already been executed successfully in a sandbox (no runtime errors).
Your job is to review it for everything beyond mere executability:

  1. Correctness   — Does it fully satisfy the task requirement?
  2. Completeness  — Are edge cases handled? Is output correct and complete?
  3. Code quality  — Is it readable, modular, and idiomatic?
  4. Security      — Any injection, unchecked input, or credential exposure?
  5. Robustness    — Does it handle unexpected input gracefully?

Be strict. 'passed' means you would confidently ship this code to production.
If you find ANY issue — even minor — set verdict to 'failed' and list each
problem as a separate, specific item in `issues`.
"""


def _count_failures(memory: list[str], task_id: str) -> int:
    """Count how many QA failure entries exist for `task_id` in global_memory."""
    marker = f"[QA — {task_id}] FAILED"
    return sum(1 for e in memory if e.startswith(marker))


def _build_failure_return(
    reviewer_task: Task,
    coder_task: Task,
    attempt: int,
    failure_reason: str,
    allow_retry: bool,
) -> dict[str, Any]:
    """Construct the state update for a failed QA cycle."""
    suffix = "" if allow_retry else " — MAX RETRIES REACHED. Halting."
    memory_entry = (
        f"[QA — {coder_task.id}] FAILED "
        f"(attempt {attempt}/{_MAX_RETRIES}). "
        f"{failure_reason}{suffix}"
    )
    task_updates: list[Task] = [
        reviewer_task.model_copy(update={"status": "failed", "output": failure_reason}),
        coder_task.model_copy(
            update={
                # "pending" signals the router to retry; "failed" signals to halt.
                "status": "pending" if allow_retry else "failed",
                "output": None if allow_retry else coder_task.output,
            }
        ),
    ]
    return {
        "tasks_list": task_updates,
        "global_memory": [memory_entry],
        "current_agent": "reviewer",
    }


def qa_agent(state: AgentState) -> dict[str, Any]:
    """Two-phase QA node:

    Phase 1 — Sandbox execution
        Runs the coder's output in a subprocess. A runtime error (syntax error,
        exception, timeout) is captured verbatim and fed back to the coder as
        an exact error log — no LLM call is made.

    Phase 2 — LLM review (only if execution passes)
        Checks correctness, completeness, quality, and security. If issues are
        found the feedback is fed back to the coder.

    Retry limit: _MAX_RETRIES total attempts. On the final attempt the coder
    task is set to 'failed' so the router stops retrying.
    """
    reviewer_task = next(
        (t for t in state["tasks_list"]
         if t.assigned_agent == "reviewer" and t.status != "completed"),
        None,
    )
    if reviewer_task is None:
        return {}

    coder_task = next(
        (t for t in reversed(state["tasks_list"])
         if t.assigned_agent == "coder" and t.status == "completed" and t.output),
        None,
    )
    if coder_task is None:
        return {}

    attempt     = _count_failures(state["global_memory"], coder_task.id) + 1
    allow_retry = attempt < _MAX_RETRIES

    # ── Phase 1: execute in sandbox ───────────────────────────────────────────
    exec_ok, exec_output = _execute_code(coder_task.output)

    if not exec_ok:
        return _build_failure_return(
            reviewer_task,
            coder_task,
            attempt,
            failure_reason=(
                f"Execution failed with the following error — fix it exactly:\n\n"
                f"```\n{exec_output}\n```"
            ),
            allow_retry=allow_retry,
        )

    # ── Phase 2: LLM static review ────────────────────────────────────────────
    structured_llm = _llm().with_structured_output(_QAOutput)

    result: _QAOutput = structured_llm.invoke([
        SystemMessage(content=_QA_SYSTEM),
        HumanMessage(
            content=(
                f"Original task requirement:\n{coder_task.description}\n\n"
                f"Code to review:\n```python\n{coder_task.output}\n```\n\n"
                f"Sandbox execution output:\n{exec_output}"
            )
        ),
    ])

    if result.verdict == "passed":
        return {
            "tasks_list": [
                reviewer_task.model_copy(update={"status": "completed", "output": result.feedback}),
            ],
            "global_memory": [f"[QA — {coder_task.id}] PASSED. {result.feedback}"],
            "current_agent": "reviewer",
        }

    issues_text = " | ".join(result.issues)
    return _build_failure_return(
        reviewer_task,
        coder_task,
        attempt,
        failure_reason=(
            f"Static review issues: {issues_text}. "
            f"Reviewer feedback: {result.feedback}"
        ),
        allow_retry=allow_retry,
    )
