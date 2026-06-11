"""
Knowledge Index — fast registry of every `.knowledge.yaml` on disk.

Keeps a lightweight SQLite table at `~/.npcsh/knowledge_index.db` that maps
memory_id → (directory_path, file_mtime) so cross-directory searches don’t
need to walk the filesystem.

The indexer is updated lazily:
  - on write (append_memory / append_link) we upsert the directory row
  - on search we validate mtime and skip stale entries
  - on explicit scan we crawl a root and rebuild
"""

import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

INDEX_DB_DIR = os.path.expanduser("~/.npcsh")
INDEX_DB_PATH = os.path.join(INDEX_DB_DIR, "knowledge_index.db")

_lock = threading.Lock()


def _ensure_index_db():
    os.makedirs(INDEX_DB_DIR, exist_ok=True)
    conn = sqlite3.connect(INDEX_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_files (
            directory TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            memory_count INTEGER DEFAULT 0,
            link_count INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def upsert_directory(directory: str, memory_count: int = 0, link_count: int = 0):
    _ensure_index_db()
    file_path = os.path.join(directory, ".knowledge.yaml")
    mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
    with _lock:
        conn = sqlite3.connect(INDEX_DB_PATH)
        conn.execute(
            """INSERT INTO knowledge_files
               (directory, mtime, memory_count, link_count, last_updated)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(directory) DO UPDATE SET
               mtime=excluded.mtime,
               memory_count=excluded.memory_count,
               link_count=excluded.link_count,
               last_updated=excluded.last_updated""",
            (directory, mtime, memory_count, link_count, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()


def remove_directory(directory: str):
    _ensure_index_db()
    with _lock:
        conn = sqlite3.connect(INDEX_DB_PATH)
        conn.execute("DELETE FROM knowledge_files WHERE directory=?", (directory,))
        conn.commit()
        conn.close()


def get_known_directories(min_mtime: float = None) -> List[Dict[str, Any]]:
    _ensure_index_db()
    conn = sqlite3.connect(INDEX_DB_PATH)
    cursor = conn.cursor()
    if min_mtime is not None:
        cursor.execute(
            "SELECT directory, mtime, memory_count, link_count FROM knowledge_files WHERE mtime >= ?",
            (min_mtime,),
        )
    else:
        cursor.execute(
            "SELECT directory, mtime, memory_count, link_count FROM knowledge_files"
        )
    rows = [
        {"directory": r[0], "mtime": r[1], "memory_count": r[2], "link_count": r[3]}
        for r in cursor.fetchall()
    ]
    conn.close()
    return rows


def scan_root(root: str, max_depth: int = 5) -> List[str]:
    """Walk `root` and register every directory containing `.knowledge.yaml`."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if ".knowledge.yaml" in filenames:
            rel = os.path.relpath(dirpath, root).split(os.sep)
            if len(rel) <= max_depth:
                found.append(dirpath)
    # batch upsert
    _ensure_index_db()
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = sqlite3.connect(INDEX_DB_PATH)
        for d in found:
            fp = os.path.join(d, ".knowledge.yaml")
            mtime = os.path.getmtime(fp) if os.path.exists(fp) else 0
            # quick count
            try:
                import yaml
                with open(fp, "r") as f:
                    data = yaml.safe_load(f) or {}
                mem_count = len(data.get("memories", []))
                link_count = len(data.get("knowledge", []))
            except Exception:
                mem_count = 0
                link_count = 0
            conn.execute(
                """INSERT INTO knowledge_files
                   (directory, mtime, memory_count, link_count, last_updated)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(directory) DO UPDATE SET
                   mtime=excluded.mtime,
                   memory_count=excluded.memory_count,
                   link_count=excluded.link_count,
                   last_updated=excluded.last_updated""",
                (d, mtime, mem_count, link_count, now),
            )
        conn.commit()
        conn.close()
    return found
