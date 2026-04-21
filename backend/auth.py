"""
auth.py — User authentication and per-user history using SQLite.

Tables:
  users        — id, username, password_hash, created_at
  chat_history — id, user_id, doc_name, role, content, sources, timestamp
  generations  — id, user_id, doc_name, gen_type, params, output, timestamp

FIX: Added delete_generation() and delete_generations_by_type() so users
can delete individual items or clear all questions/summaries from History tab.
"""

import sqlite3
import hashlib
import os
import json
from datetime import datetime

DB_PATH = "edify_users.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    conn.cursor().executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            created_at  TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            doc_name    TEXT    NOT NULL,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            sources     TEXT    DEFAULT '[]',
            timestamp   TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS generations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            doc_name    TEXT    NOT NULL,
            gen_type    TEXT    NOT NULL,
            params      TEXT    DEFAULT '{}',
            output      TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()


# ─── Auth ─────────────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username: str, password: str) -> tuple[bool, str]:
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
            (username.strip().lower(), _hash(password), datetime.now().isoformat())
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already taken."
    finally:
        conn.close()


def login_user(username: str, password: str) -> tuple[bool, dict | None, str]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username.strip().lower(), _hash(password))
    ).fetchone()
    conn.close()
    if row:
        return True, dict(row), "Login successful."
    return False, None, "Invalid username or password."


# ─── Chat History ─────────────────────────────────────────────────────────────

def save_chat_message(user_id, doc_name, role, content, sources):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO chat_history (user_id, doc_name, role, content, sources, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, doc_name, role, content, json.dumps(sources), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def load_chat_history(user_id, doc_name) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content, sources, timestamp FROM chat_history "
        "WHERE user_id = ? AND doc_name = ? ORDER BY id ASC",
        (user_id, doc_name)
    ).fetchall()
    conn.close()
    return [{
        "role": r["role"], "content": r["content"],
        "sources": json.loads(r["sources"] or "[]"),
        "timestamp": r["timestamp"],
    } for r in rows]


def get_all_chat_docs(user_id) -> list[str]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT DISTINCT doc_name FROM chat_history WHERE user_id = ? ORDER BY doc_name",
        (user_id,)
    ).fetchall()
    conn.close()
    return [r["doc_name"] for r in rows]


def delete_chat_history(user_id, doc_name):
    """Delete all chat messages for a user+document pair."""
    conn = _get_conn()
    conn.execute(
        "DELETE FROM chat_history WHERE user_id = ? AND doc_name = ?",
        (user_id, doc_name)
    )
    conn.commit()
    conn.close()


def delete_all_chat_history(user_id):
    """Delete ALL chat history for a user."""
    conn = _get_conn()
    conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


# ─── Generations ──────────────────────────────────────────────────────────────

def save_generation(user_id, doc_name, gen_type, params, output):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO generations (user_id, doc_name, gen_type, params, output, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, doc_name, gen_type, json.dumps(params), output, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def load_generations(user_id, gen_type=None) -> list[dict]:
    conn = _get_conn()
    if gen_type:
        rows = conn.execute(
            "SELECT * FROM generations WHERE user_id = ? AND gen_type = ? ORDER BY id DESC",
            (user_id, gen_type)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM generations WHERE user_id = ? ORDER BY id DESC",
            (user_id,)
        ).fetchall()
    conn.close()
    return [{
        "id": r["id"], "doc_name": r["doc_name"], "gen_type": r["gen_type"],
        "params": json.loads(r["params"] or "{}"),
        "output": r["output"], "timestamp": r["timestamp"],
    } for r in rows]


def delete_generation(user_id, gen_id):
    """Delete a single generation record (verified by user_id for security)."""
    conn = _get_conn()
    conn.execute(
        "DELETE FROM generations WHERE id = ? AND user_id = ?",
        (gen_id, user_id)
    )
    conn.commit()
    conn.close()


def delete_generations_by_type(user_id, gen_type):
    """Delete all generations of a given type for a user."""
    conn = _get_conn()
    conn.execute(
        "DELETE FROM generations WHERE user_id = ? AND gen_type = ?",
        (user_id, gen_type)
    )
    conn.commit()
    conn.close()


def delete_all_generations(user_id):
    """Delete ALL generation records for a user."""
    conn = _get_conn()
    conn.execute("DELETE FROM generations WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


# ─── Stats ─────────────────────────────────────────────────────────────────────

def get_user_stats(user_id) -> dict:
    conn = _get_conn()
    chat_count = conn.execute(
        "SELECT COUNT(*) FROM chat_history WHERE user_id = ? AND role = 'user'", (user_id,)
    ).fetchone()[0]
    q_count = conn.execute(
        "SELECT COUNT(*) FROM generations WHERE user_id = ? AND gen_type = 'questions'", (user_id,)
    ).fetchone()[0]
    s_count = conn.execute(
        "SELECT COUNT(*) FROM generations WHERE user_id = ? AND gen_type = 'summary'", (user_id,)
    ).fetchone()[0]
    doc_count = conn.execute(
        "SELECT COUNT(DISTINCT doc_name) FROM chat_history WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    conn.close()
    return {
        "chat_turns": chat_count, "question_sets": q_count,
        "summaries": s_count,     "docs_used": doc_count,
    }
