"""
Streamlit visual interface for the multi-agent LangGraph workflow.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Workflow",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Display constants ─────────────────────────────────────────────────────────
_STATUS_LABEL: dict[str, str] = {
    "pending":     "⏳ Pending",
    "in_progress": "⚙️  Running",
    "completed":   "✅ Done",
    "failed":      "❌ Failed",
}

_AGENT_ICON: dict[str, str] = {
    "orchestrator": "🧠",
    "researcher":   "🔍",
    "analyst":      "📊",
    "writer":       "✍️ ",
    "coder":        "💻",
    "reviewer":     "🔬",
}

_MEMORY_ICON: dict[str, str] = {
    "[Research": "🔍",
    "[Code":     "💻",
    "[QA":       "🔬",
    "Orchestrat": "🧠",
}

_NODE_LABEL: dict[str, str] = {
    "orchestrator": "Orchestrator",
    "research":     "Research Agent",
    "coder":        "Coder Agent",
    "qa":           "QA Agent (Critic)",
}

# ── Session-state defaults ────────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "tasks_list":    [],
    "global_memory": [],
    "step_log":      [],
    "is_running":    False,
    "workflow_done": False,
    "current_node":  "",
    "error":         None,
    "attempt_map":   {},   # task_id → retry count, for UI badge
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── State helpers ─────────────────────────────────────────────────────────────

def _merge_tasks(existing: list, updates: list) -> list:
    index = {t.id: t for t in existing}
    for t in updates:
        index[t.id] = t
    return list(index.values())


def _apply_update(update: dict[str, Any]) -> None:
    """Mirror LangGraph reducers into session_state."""
    for key, value in update.items():
        if key == "tasks_list":
            st.session_state.tasks_list = _merge_tasks(
                st.session_state.tasks_list, value
            )
        elif key == "global_memory":
            st.session_state.global_memory += value
        else:
            st.session_state[key] = value


# ── DataFrame builder ─────────────────────────────────────────────────────────

def _tasks_dataframe(tasks: list) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tasks, 1):
        icon        = _AGENT_ICON.get(t.assigned_agent, "⚙️")
        status      = _STATUS_LABEL.get(t.status, t.status)
        description = t.description if len(t.description) <= 90 else t.description[:90] + "…"
        output      = ""
        if t.output:
            line   = t.output.splitlines()[0]
            output = line if len(line) <= 75 else line[:75] + "…"

        retries = st.session_state.attempt_map.get(t.id, 0)
        retry_badge = f"  🔄×{retries}" if retries else ""

        rows.append({
            "#":           i,
            "Agent":       f"{icon} {t.assigned_agent}",
            "Description": description,
            "Status":      status + retry_badge,
            "Output":      output,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["#", "Agent", "Description", "Status", "Output"]
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar(placeholder: "st.delta_generator.DeltaGenerator") -> None:
    memory: list[str] = st.session_state.global_memory
    with placeholder.container():
        if not memory:
            st.caption("Memory is empty — entries appear as agents run.")
            return
        for idx, entry in enumerate(reversed(memory), 1):
            icon  = next((ic for pfx, ic in _MEMORY_ICON.items() if entry.startswith(pfx)), "📝")
            clean = entry.split("] ", 1)[-1] if "] " in entry else entry
            with st.container():
                st.markdown(
                    f"<div style='font-size:0.82rem; line-height:1.5'>"
                    f"<span style='opacity:0.5'>#{len(memory) - idx + 1}</span>&nbsp;"
                    f"{icon}&nbsp;{clean}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.divider()


with st.sidebar:
    st.markdown("## 🧠 Global Memory")
    st.caption("Key insights accumulated across all agents")
    st.divider()
    sidebar_placeholder = st.empty()


# ── Main layout ───────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='margin-bottom:0'>🤖 Multi-Agent AI Workflow</h1>",
    unsafe_allow_html=True,
)
st.caption("Powered by LangGraph · Claude 3.7 Sonnet · Supabase")
st.divider()

# Objective input row
input_col, btn_col, reset_col = st.columns([5, 1, 1])

with input_col:
    objective = st.text_input(
        "Objective",
        label_visibility="collapsed",
        placeholder="e.g. Write a Python web scraper that fetches top Hacker News posts",
        disabled=st.session_state.is_running,
    )

with btn_col:
    run_clicked = st.button(
        "▶ Run",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_running or not bool((objective or "").strip()),
    )

with reset_col:
    if st.button("↺ Reset", use_container_width=True, disabled=st.session_state.is_running):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

# Metrics strip (only shown once tasks exist)
metrics_placeholder = st.empty()

# Current-step status widget
step_status_placeholder = st.empty()

st.markdown("### 📋 Task Pipeline")
table_placeholder = st.empty()

st.markdown("### 📜 Execution Log")
log_placeholder = st.empty()


# ── Render helpers (called on every script run + inside streaming loop) ───────

def _render_metrics() -> None:
    tasks = st.session_state.tasks_list
    if not tasks:
        metrics_placeholder.empty()
        return
    total     = len(tasks)
    completed = sum(1 for t in tasks if t.status == "completed")
    failed    = sum(1 for t in tasks if t.status == "failed")
    pending   = sum(1 for t in tasks if t.status == "pending")
    running   = sum(1 for t in tasks if t.status == "in_progress")

    with metrics_placeholder.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total",     total)
        c2.metric("✅ Done",   completed)
        c3.metric("⚙️  Running", running)
        c4.metric("⏳ Pending", pending)
        c5.metric("❌ Failed",  failed)


def _render_table() -> None:
    with table_placeholder.container():
        df = _tasks_dataframe(st.session_state.tasks_list)
        if df.empty:
            st.caption("Tasks appear here once the Orchestrator generates a plan…")
        else:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "#":           st.column_config.NumberColumn(width="small"),
                    "Agent":       st.column_config.TextColumn(width="medium"),
                    "Description": st.column_config.TextColumn(width="large"),
                    "Status":      st.column_config.TextColumn(width="medium"),
                    "Output":      st.column_config.TextColumn("Output Preview", width="large"),
                },
            )


def _render_log() -> None:
    with log_placeholder.container():
        if not st.session_state.step_log:
            st.caption("Execution steps appear here…")
            return
        for entry in reversed(st.session_state.step_log):
            st.markdown(entry, unsafe_allow_html=True)


# Paint current state on every non-streaming render
_render_sidebar(sidebar_placeholder)
_render_metrics()
_render_table()
_render_log()

if st.session_state.error:
    st.error(f"**Workflow error:** {st.session_state.error}")

if st.session_state.workflow_done and not st.session_state.is_running:
    total = len(st.session_state.tasks_list)
    done  = sum(1 for t in st.session_state.tasks_list if t.status == "completed")
    st.success(f"✅ Workflow complete — {done}/{total} tasks succeeded.")


# ── Workflow execution ────────────────────────────────────────────────────────

if run_clicked and (objective or "").strip() and not st.session_state.is_running:

    # Reset
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.is_running = True

    initial_state: dict[str, Any] = {
        "user_objective": objective.strip(),
        "tasks_list":     [],
        "global_memory":  [],
        "current_agent":  "",
    }

    from src.graphs.agent_graph import get_workflow

    step = 0

    try:
        for chunk in get_workflow().stream(initial_state, stream_mode="updates"):
            for node_name, update in chunk.items():
                if node_name.startswith("__"):
                    continue

                step += 1
                node_icon  = _AGENT_ICON.get(node_name, "⚙️")
                node_label = _NODE_LABEL.get(node_name, node_name.replace("_", " ").title())
                ts         = time.strftime("%H:%M:%S")

                # Track coder retries in attempt_map for the UI badge
                if node_name == "qa":
                    for t in update.get("tasks_list", []):
                        if t.assigned_agent == "coder" and t.status == "pending":
                            st.session_state.attempt_map[t.id] = (
                                st.session_state.attempt_map.get(t.id, 0) + 1
                            )

                _apply_update(update)

                # ── Live UI updates ───────────────────────────────────────────

                with step_status_placeholder.container():
                    st.info(
                        f"{node_icon} **Step {step} — {node_label}** completed",
                        icon="✔️",
                    )

                _render_metrics()
                _render_table()
                _render_sidebar(sidebar_placeholder)

                # Build log line
                tasks_in_update = update.get("tasks_list", [])
                task_note = ""
                if tasks_in_update:
                    t0 = tasks_in_update[0]
                    label = _STATUS_LABEL.get(t0.status, t0.status)
                    task_note = (
                        f"&nbsp;·&nbsp;<span style='opacity:0.7'>"
                        f"{t0.assigned_agent}: {t0.description[:55]}… → {label}"
                        f"</span>"
                    )

                qa_retry = (
                    node_name == "qa"
                    and any(
                        t.assigned_agent == "coder" and t.status == "pending"
                        for t in update.get("tasks_list", [])
                    )
                )
                retry_badge = "&nbsp;🔄 <em>retrying coder…</em>" if qa_retry else ""

                st.session_state.step_log.append(
                    f"<div style='font-size:0.85rem; padding:2px 0'>"
                    f"<span style='opacity:0.5'>{ts}</span>&nbsp;"
                    f"<strong>{node_icon} {node_label}</strong>"
                    f"{task_note}{retry_badge}"
                    f"</div>"
                )
                _render_log()

        st.session_state.workflow_done = True

    except Exception as exc:
        st.session_state.error = str(exc)

    finally:
        st.session_state.is_running = False

        with step_status_placeholder.container():
            if st.session_state.workflow_done:
                st.success("✅ All agents finished.")
            elif st.session_state.error:
                st.error(f"❌ {st.session_state.error}")

    # Supabase save
    if st.session_state.workflow_done:
        try:
            from src.memory.database import save_run
            with st.spinner("Saving run to Supabase…"):
                row = save_run(
                    objective=objective.strip(),
                    tasks=st.session_state.tasks_list,
                    memory=st.session_state.global_memory,
                )
            short_id = str(row.get("id", ""))[:8]
            st.toast(f"🗄️ Saved to Supabase (id: {short_id}…)", icon="✅")
        except EnvironmentError:
            st.toast("⚠️ Supabase not configured — run not persisted.", icon="⚠️")
        except Exception as exc:
            st.toast(f"⚠️ Supabase save failed: {exc}", icon="⚠️")

    st.rerun()
