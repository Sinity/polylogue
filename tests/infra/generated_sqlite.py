"""Neutral SQLite shapes with acquired generated values."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

GeneratedStorage = Literal["VIRTUAL", "STORED"]


def generated_thread_state(path: Path, storage: GeneratedStorage) -> Path:
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript(f"""
            CREATE TABLE threads (
                id TEXT PRIMARY KEY, title_seed TEXT,
                title TEXT GENERATED ALWAYS AS (title_seed || ' curated') {storage},
                cwd TEXT, created_at_ms INTEGER, updated_at_ms INTEGER,
                source TEXT, model TEXT, agent_nickname TEXT, agent_role TEXT,
                archived INTEGER
            );
            CREATE TABLE thread_spawn_edges (
                parent_thread_id TEXT, child_thread_id TEXT, status TEXT
            );
            INSERT INTO threads(rowid,id,title_seed,cwd,created_at_ms,updated_at_ms,source,archived)
            VALUES (17,'thread-1','neutral','/synthetic',1,2,'cli',0);
        """)
    return path


def generated_state_parts(path: Path, storage: GeneratedStorage, *, kind: str) -> Path:
    with closing(sqlite3.connect(path)) as conn, conn:
        if kind == "goals":
            conn.execute(f"""CREATE TABLE thread_goals (
                thread_id TEXT, goal_id TEXT, seed TEXT,
                objective TEXT GENERATED ALWAYS AS (seed) {storage}
            )""").close()
            conn.execute(
                "INSERT INTO thread_goals(rowid,thread_id,goal_id,seed) VALUES (19,'thread-1','goal-1',?)",
                ("intent Ω\x00 retained",),
            ).close()
        else:
            conn.execute(f"""CREATE TABLE stage1_outputs (
                thread_id TEXT, seed TEXT, summary_seed TEXT,
                raw_memory TEXT GENERATED ALWAYS AS (seed) {storage},
                rollout_summary TEXT GENERATED ALWAYS AS (summary_seed) {storage},
                source_updated_at INTEGER, generated_at INTEGER, usage_count INTEGER, selected_for_phase2 INTEGER
            )""").close()
            conn.execute(
                "INSERT INTO stage1_outputs(rowid,thread_id,seed,summary_seed,source_updated_at,generated_at,usage_count,selected_for_phase2) VALUES (23,'thread-1',?,?,1,2,0,0)",
                ("memory Ω\x00 retained", "summary retained"),
            ).close()
    return path
