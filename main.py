import sys
import time
from typing import Any, Iterator

from dotenv import load_dotenv
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

load_dotenv()

console = Console(highlight=False)

# ── Theme ─────────────────────────────────────────────────────────────────────

_C = {
    "orchestrator": "magenta",
    "research":     "cyan",
    "coder":        "yellow",
    "qa":           "red",
    "success":      "green",
    "warning":      "yellow",
    "muted":        "bright_black",
    "accent":       "bright_blue",
}

_AGENT_META: dict[str, dict[str, str]] = {
    "orchestrator": {"icon": "🧠", "label": "Orchestrator",      "color": _C["orchestrator"]},
    "research":     {"icon": "🔍", "label": "Research Agent",    "color": _C["research"]},
    "coder":        {"icon": "💻", "label": "Coder Agent",       "color": _C["coder"]},
    "qa":           {"icon": "🔬", "label": "QA Agent (Critic)", "color": _C["qa"]},
}

_STATUS_STYLE: dict[str, str] = {
    "pending":     "yellow",
    "in_progress": "cyan",
    "completed":   "bold green",
    "failed":      "bold red",
}

_STATUS_ICON: dict[str, str] = {
    "pending":     "⏳",
    "in_progress": "⚙️ ",
    "completed":   "✅",
    "failed":      "❌",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_language(code: str) -> str:
    """Best-effort language detection for syntax highlighting."""
    first = code.lstrip()
    if first.startswith("<!DOCTYPE") or first.startswith("<html"):
        return "html"
    if first.startswith("{") or first.startswith("["):
        return "json"
    if "def " in code or "import " in code or "class " in code:
        return "python"
    if "function " in code or "const " in code or "=>" in code:
        return "javascript"
    if "#include" in code or "int main" in code:
        return "cpp"
    if "SELECT " in code.upper() or "CREATE TABLE" in code.upper():
        return "sql"
    return "text"


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def _merge_state(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Mirror the LangGraph reducer logic for local display tracking."""
    result = dict(base)
    for key, value in update.items():
        if key == "tasks_list":
            index = {t.id: t for t in result.get("tasks_list", [])}
            for t in value:
                index[t.id] = t
            result["tasks_list"] = list(index.values())
        elif key == "global_memory":
            result["global_memory"] = result.get("global_memory", []) + value
        else:
            result[key] = value
    return result


# ── Step renderers ────────────────────────────────────────────────────────────

def _render_orchestrator(update: dict[str, Any]) -> None:
    tasks = update.get("tasks_list", [])
    if not tasks:
        return

    tree = Tree(
        f"[bold {_C['orchestrator']}]📋  {len(tasks)} task{'s' if len(tasks) != 1 else ''} planned[/]",
        guide_style=_C["muted"],
    )

    agent_icons = {"researcher": "🔍", "coder": "💻", "reviewer": "🔬",
                   "analyst": "📊", "writer": "✍️ "}

    for i, task in enumerate(tasks, 1):
        icon = agent_icons.get(task.assigned_agent, "⚙️ ")
        branch = tree.add(
            f"[{_C['muted']}]{i}.[/]  {icon} "
            f"[bold]{task.assigned_agent}[/]  "
            f"[{_C['muted']}]→[/]  {task.description}"
        )

    memory = update.get("global_memory", [])
    reasoning = next((m for m in memory if m.startswith("Orchestrator reasoning:")), None)
    if reasoning:
        note = reasoning.replace("Orchestrator reasoning: ", "")
        tree.add(f"[{_C['muted']}italic]💭 {note}[/]")

    console.print(
        Panel(tree, title=f"[bold {_C['orchestrator']}]Task Plan[/]",
              border_style=_C["orchestrator"], padding=(1, 2))
    )


def _render_research(update: dict[str, Any]) -> None:
    entries: list[str] = update.get("global_memory", [])
    tasks: list[Any]   = update.get("tasks_list", [])

    research_entries = [e for e in entries if e.startswith("[Research")]
    if not research_entries:
        return

    # Insights as markdown bullet list
    md_lines = ["## 🔍 Insights Gathered\n"]
    for entry in research_entries:
        # Strip the "[Research — <uuid>] " prefix for display
        clean = entry.split("] ", 1)[-1] if "] " in entry else entry
        md_lines.append(f"- {clean}")

    console.print(
        Panel(
            Markdown("\n".join(md_lines)),
            title=f"[bold {_C['research']}]Research[/]",
            border_style=_C["research"],
            padding=(1, 2),
        )
    )

    for task in tasks:
        if task.assigned_agent == "researcher" and task.status == "completed" and task.output:
            console.print(
                Panel(
                    Markdown(f"**Summary**\n\n{task.output}"),
                    title=f"[{_C['muted']}]Research Summary[/]",
                    border_style=_C["muted"],
                    padding=(0, 2),
                )
            )


def _render_coder(update: dict[str, Any]) -> None:
    tasks:  list[Any] = update.get("tasks_list", [])
    memory: list[str] = update.get("global_memory", [])

    for task in tasks:
        if task.assigned_agent != "coder" or task.status != "completed" or not task.output:
            continue

        code   = task.output
        lang   = _detect_language(code)
        lines  = code.splitlines()
        limit  = 50
        shown  = "\n".join(lines[:limit])

        syntax = Syntax(
            shown,
            lang,
            theme="monokai",
            line_numbers=True,
            word_wrap=False,
        )

        console.print(
            Panel(
                syntax,
                title=f"[bold {_C['coder']}]Generated Code  [{_C['muted']}]{lang}  ·  {len(lines)} lines[/][/]",
                border_style=_C["coder"],
                padding=(0, 1),
            )
        )

        if len(lines) > limit:
            console.print(
                f"  [{_C['muted']}]… {len(lines) - limit} more lines not shown[/]"
            )

    for entry in memory:
        if entry.startswith("[Code"):
            note = entry.split("] ", 1)[-1] if "] " in entry else entry
            console.print(
                f"  [{_C['muted']}]💡 Design note:[/] [{_C['coder']}]{note}[/]"
            )


def _render_qa(update: dict[str, Any]) -> None:
    entries: list[str] = update.get("global_memory", [])
    tasks:   list[Any] = update.get("tasks_list", [])

    qa_entry = next((e for e in entries if "[QA —" in e), None)
    if not qa_entry:
        return

    passed = "PASSED" in qa_entry

    # Verdict banner
    if passed:
        verdict = Text.assemble(
            ("  ✅  VERDICT: PASSED  ", f"bold {_C['success']}"),
        )
        border = _C["success"]
    else:
        verdict = Text.assemble(
            ("  ❌  VERDICT: FAILED  ", f"bold {_C['qa']}"),
        )
        border = _C["qa"]

    console.print(
        Panel(
            Align.center(verdict),
            title=f"[bold {_C['qa']}]QA Review[/]",
            border_style=border,
            padding=(1, 2),
        )
    )

    # Feedback block
    for task in tasks:
        if task.assigned_agent == "reviewer" and task.output:
            icon = "✅" if passed else "🔄"
            action = "Approved for delivery." if passed else "Returning to Coder for revision."
            console.print(
                Panel(
                    Markdown(f"{task.output}\n\n---\n*{icon} {action}*"),
                    title=f"[{_C['muted']}]Reviewer Feedback[/]",
                    border_style=border,
                    padding=(0, 2),
                )
            )


_RENDERERS = {
    "orchestrator": _render_orchestrator,
    "research":     _render_research,
    "coder":        _render_coder,
    "qa":           _render_qa,
}


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(state: dict[str, Any]) -> None:
    tasks  = state.get("tasks_list", [])
    memory = state.get("global_memory", [])
    done   = sum(1 for t in tasks if t.status == "completed")
    total  = len(tasks)

    console.print()
    console.print(Rule(f"[bold white]Workflow Complete  [{_C['muted']}]{done}/{total} tasks succeeded[/][/]"))
    console.print()

    # ── Task table ────────────────────────────────────────────────────────────
    table = Table(
        box=box.HEAVY_OUTLINE,
        show_header=True,
        header_style=f"bold white on {_C['accent']}",
        expand=True,
        padding=(0, 1),
        border_style=_C["accent"],
    )
    table.add_column("",           width=3,  justify="center")
    table.add_column("Agent",      width=13, style="bold")
    table.add_column("Task",       ratio=3)
    table.add_column("Status",     width=13, justify="center")
    table.add_column("Output preview", ratio=2, style=_C["muted"])

    for i, task in enumerate(tasks, 1):
        icon   = _STATUS_ICON.get(task.status, "?")
        style  = _STATUS_STYLE.get(task.status, "white")
        output = ""
        if task.output:
            first = task.output.splitlines()[0]
            output = _truncate(first, 55)
        table.add_row(
            f"[{_C['muted']}]{i}[/]",
            task.assigned_agent,
            _truncate(task.description, 70),
            Text(f"{icon} {task.status}", style=style),
            output,
        )

    console.print(table)

    # ── Memory log ────────────────────────────────────────────────────────────
    if memory:
        console.print()
        console.print(Rule(f"[bold {_C['research']}]Global Memory Log  [{_C['muted']}]{len(memory)} entries[/][/]"))
        console.print()

        category_icons = {
            "[Research": ("🔍", _C["research"]),
            "[Code":     ("💻", _C["coder"]),
            "[QA":       ("🔬", _C["qa"]),
            "Orchestrat": ("🧠", _C["orchestrator"]),
        }

        for i, entry in enumerate(memory, 1):
            icon, color = next(
                ((ic, co) for prefix, (ic, co) in category_icons.items() if entry.startswith(prefix)),
                ("📝", "white"),
            )
            label = f"[{_C['muted']}]{i:>3}.[/]  {icon}  "
            clean = entry.split("] ", 1)[-1] if "] " in entry else entry
            console.print(f"{label}[{color}]{clean}[/]")


# ── Supabase ──────────────────────────────────────────────────────────────────

def _save_to_supabase(state: dict[str, Any], objective: str) -> None:
    console.print()
    console.print(Rule(f"[bold {_C['success']}]Saving to Supabase[/]"))
    console.print()

    try:
        from src.graphs.agent_graph import Task
        from src.memory.database import save_run

        tasks  = state.get("tasks_list", [])
        memory = state.get("global_memory", [])

        with console.status("[dim]Writing to database…[/]", spinner="dots"):
            row = save_run(objective=objective, tasks=tasks, memory=memory)

        info = Table.grid(padding=(0, 2))
        info.add_column(style=_C["muted"])
        info.add_column()
        info.add_row("run id",     str(row.get("id", "n/a")))
        info.add_row("created at", str(row.get("created_at", "n/a")))
        info.add_row("tasks",      str(row.get("task_count", len(tasks))))
        info.add_row("memory",     f"{len(memory)} entries")

        console.print(
            Panel(
                info,
                title=f"[bold {_C['success']}]✓  Saved[/]",
                border_style=_C["success"],
                padding=(1, 2),
            )
        )

    except EnvironmentError as exc:
        console.print(
            Panel(
                f"[{_C['warning']}]⚠  Skipped — {exc}[/]\n"
                f"[{_C['muted']}]Set SUPABASE_URL and SUPABASE_KEY in .env to enable persistence.[/]",
                border_style=_C["warning"],
                padding=(0, 2),
            )
        )
    except Exception as exc:
        console.print(
            Panel(
                f"[{_C['qa']}]✗  Save failed:[/] {exc}",
                border_style=_C["qa"],
                padding=(0, 2),
            )
        )


# ── Stream loop ───────────────────────────────────────────────────────────────

def _stream_with_spinner(
    initial_state: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Drive workflow.stream() with a spinner between node executions.
    Yields (node_name, update_dict) for every non-internal step."""
    from src.graphs.agent_graph import get_workflow

    stream = iter(get_workflow().stream(initial_state, stream_mode="updates"))

    while True:
        # Show spinner while the next node is running
        with console.status(
            f"[{_C['muted']}]Agent working…[/]",
            spinner="dots",
            spinner_style=_C["accent"],
        ):
            try:
                chunk = next(stream)
            except StopIteration:
                return

        for node_name, update in chunk.items():
            if not node_name.startswith("__"):
                yield node_name, update


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print()

    # ── Banner ────────────────────────────────────────────────────────────────
    banner = Align.center(
        Panel.fit(
            Align.center(
                f"[bold white]🤖  Multi-Agent AI Workflow[/]\n"
                f"[{_C['muted']}]LangGraph  ·  Claude 3.7 Sonnet  ·  Supabase[/]"
            ),
            border_style=_C["accent"],
            padding=(1, 6),
        )
    )
    console.print(banner)
    console.print()

    # ── Objective input ───────────────────────────────────────────────────────
    try:
        objective = console.input(
            f"[bold {_C['accent']}]  ▶  Objective:[/]  "
        ).strip()
    except (KeyboardInterrupt, EOFError):
        console.print(f"\n  [{_C['muted']}]Aborted.[/]")
        sys.exit(0)

    if not objective:
        console.print(f"  [{_C['qa']}]✗  Objective cannot be empty.[/]")
        sys.exit(1)

    console.print()
    console.print(
        Panel(
            f"[italic white]{objective}[/]",
            title=f"[bold {_C['accent']}]Objective[/]",
            border_style=_C["accent"],
            padding=(0, 2),
        )
    )
    console.print()

    # ── Workflow stream ───────────────────────────────────────────────────────
    initial_state: dict[str, Any] = {
        "user_objective": objective,
        "tasks_list":     [],
        "global_memory":  [],
        "current_agent":  "",
    }

    step      = 0
    state     = dict(initial_state)
    timings:  list[tuple[str, float]] = []

    try:
        for node_name, update in _stream_with_spinner(initial_state):
            step += 1
            meta = _AGENT_META.get(
                node_name,
                {"icon": "⚙️ ", "label": node_name, "color": "white"},
            )

            t_start = time.perf_counter()

            # Step header
            console.print(
                Rule(
                    f"{meta['icon']}  "
                    f"[bold {meta['color']}]Step {step}  ·  {meta['label']}[/]",
                    style=meta["color"],
                )
            )
            console.print()

            renderer = _RENDERERS.get(node_name)
            if renderer:
                renderer(update)

            elapsed = time.perf_counter() - t_start
            timings.append((meta["label"], elapsed))

            state = _merge_state(state, update)
            console.print()

    except KeyboardInterrupt:
        console.print(f"\n  [{_C['warning']}]⚠  Interrupted — showing partial results.[/]\n")

    # ── Timing strip ─────────────────────────────────────────────────────────
    if timings:
        timing_cols = [
            Panel.fit(
                f"[{_C['muted']}]{label}[/]\n[bold white]{t:.1f}s[/]",
                border_style=_C["muted"],
                padding=(0, 1),
            )
            for label, t in timings
        ]
        console.print(Columns(timing_cols, equal=True, expand=True))
        console.print()

    # ── Summary + save ────────────────────────────────────────────────────────
    _print_summary(state)
    _save_to_supabase(state, objective)
    console.print()


if __name__ == "__main__":
    main()
