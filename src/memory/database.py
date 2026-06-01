import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

from src.graphs.agent_graph import Task

load_dotenv()

# Expected Supabase table DDL (run once in the SQL editor):
#
#   create table agent_runs (
#     id          uuid primary key default gen_random_uuid(),
#     objective   text        not null,
#     tasks       jsonb       not null,
#     memory      jsonb       not null,
#     task_count  int         not null,
#     created_at  timestamptz not null default now()
#   );


def _client() -> Client:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set in the environment."
        )
    return create_client(url, key)


def save_run(
    objective: str,
    tasks: list[Task],
    memory: list[str],
) -> dict[str, Any]:
    """Insert one completed workflow run into the agent_runs table.

    Returns the inserted row as a dict (includes the server-generated id
    and created_at timestamp).

    Raises EnvironmentError if credentials are missing.
    Raises an exception from the supabase client if the insert fails.
    """
    payload: dict[str, Any] = {
        "objective":  objective,
        "tasks":      [t.model_dump() for t in tasks],
        "memory":     memory,
        "task_count": len(tasks),
    }

    response = _client().table("agent_runs").insert(payload).execute()

    if not response.data:
        raise RuntimeError(f"Supabase insert returned no data: {response}")

    return response.data[0]


def fetch_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent `limit` runs, newest first."""
    response = (
        _client()
        .table("agent_runs")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []
