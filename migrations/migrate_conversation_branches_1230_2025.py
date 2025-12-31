"""
Migration to add conversation branching support.

Adds:
- branch_id column to conversation_history table
- parent_message_id column to conversation_history table
- conversation_branches table for branch metadata

Run with: python -m npcsh.db_migrations.migrate_conversation_branches_1230_2025
"""

import sqlite3
import os

def migrate(db_path=None):
    if db_path is None:
        db_path = os.path.expanduser("~/npcsh_history.db")

    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if branch_id column already exists
        cursor.execute("PRAGMA table_info(conversation_history)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'branch_id' not in columns:
            print("Adding branch_id column to conversation_history...")
            cursor.execute("ALTER TABLE conversation_history ADD COLUMN branch_id TEXT DEFAULT 'main'")
            print("Added branch_id column")
        else:
            print("branch_id column already exists")

        if 'parent_message_id' not in columns:
            print("Adding parent_message_id column to conversation_history...")
            cursor.execute("ALTER TABLE conversation_history ADD COLUMN parent_message_id TEXT")
            print("Added parent_message_id column")
        else:
            print("parent_message_id column already exists")

        # Create conversation_branches table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_branches (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                name TEXT NOT NULL,
                parent_branch_id TEXT DEFAULT 'main',
                branch_from_message_id TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        print("Created/verified conversation_branches table")

        # Create index for faster branch lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_history_branch
            ON conversation_history(conversation_id, branch_id)
        """)
        print("Created/verified branch index")

        conn.commit()
        print("Migration completed successfully!")
        return True

    except Exception as e:
        print(f"Migration failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
