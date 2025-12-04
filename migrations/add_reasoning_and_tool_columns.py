#!/usr/bin/env python3
"""
Migration script to add reasoning_content, tool_calls, and tool_results columns
to the conversation_history table.

This script adds three new columns to support:
- reasoning_content: Stores thinking tokens / chain of thought content
- tool_calls: JSON array of tool calls made by assistant
- tool_results: JSON array of tool call results

Usage:
    python add_reasoning_and_tool_columns.py [--db-path PATH]

Arguments:
    --db-path: Path to the SQLite database (default: ~/npcsh_history.db)

Example:
    python add_reasoning_and_tool_columns.py
    python add_reasoning_and_tool_columns.py --db-path /path/to/custom.db
"""

import os
import sys
import argparse
from datetime import datetime

try:
    from sqlalchemy import create_engine, text, inspect
except ImportError:
    print("ERROR: SQLAlchemy is required. Install with: pip install sqlalchemy")
    sys.exit(1)


def get_engine(db_path: str):
    """Create SQLAlchemy engine from database path"""
    if db_path.startswith('~/'):
        db_path = os.path.expanduser(db_path)

    if db_path.startswith('postgresql://') or db_path.startswith('postgres://'):
        return create_engine(db_path)
    else:
        return create_engine(f'sqlite:///{db_path}')


def column_exists(engine, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table"""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def run_migration(db_path: str, dry_run: bool = False):
    """Run the migration to add new columns"""
    print(f"Migration: Add reasoning_content, tool_calls, and tool_results columns")
    print(f"Database: {db_path}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)

    engine = get_engine(db_path)

    # Check if table exists
    inspector = inspect(engine)
    if 'conversation_history' not in inspector.get_table_names():
        print("ERROR: conversation_history table does not exist!")
        print("Please ensure the database has been initialized first.")
        return False

    # Define new columns to add
    new_columns = [
        ('reasoning_content', 'TEXT'),  # For thinking tokens / chain of thought
        ('tool_calls', 'TEXT'),         # JSON array of tool calls made by assistant
        ('tool_results', 'TEXT')        # JSON array of tool call results
    ]

    columns_added = []
    columns_skipped = []

    for column_name, column_type in new_columns:
        if column_exists(engine, 'conversation_history', column_name):
            print(f"  SKIP: Column '{column_name}' already exists")
            columns_skipped.append(column_name)
        else:
            print(f"  ADD:  Column '{column_name}' ({column_type})")
            columns_added.append((column_name, column_type))

    if not columns_added:
        print("\nNo columns need to be added. Migration already complete.")
        return True

    if dry_run:
        print("\n[DRY RUN] Would execute the following SQL:")
        for column_name, column_type in columns_added:
            print(f"  ALTER TABLE conversation_history ADD COLUMN {column_name} {column_type};")
        return True

    # Execute the migration
    print("\nExecuting migration...")

    with engine.begin() as conn:
        for column_name, column_type in columns_added:
            try:
                sql = text(f"ALTER TABLE conversation_history ADD COLUMN {column_name} {column_type}")
                conn.execute(sql)
                print(f"  SUCCESS: Added column '{column_name}'")
            except Exception as e:
                print(f"  ERROR: Failed to add column '{column_name}': {e}")
                return False

    print("\n" + "=" * 60)
    print("Migration completed successfully!")
    print(f"  Columns added: {len(columns_added)}")
    print(f"  Columns skipped (already exist): {len(columns_skipped)}")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Add reasoning_content, tool_calls, and tool_results columns to conversation_history table'
    )
    parser.add_argument(
        '--db-path',
        default='~/npcsh_history.db',
        help='Path to the SQLite database (default: ~/npcsh_history.db)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    # Expand user path
    db_path = os.path.expanduser(args.db_path)

    # Check if database file exists (for SQLite)
    if not db_path.startswith('postgresql://') and not db_path.startswith('postgres://'):
        if not os.path.exists(db_path):
            print(f"ERROR: Database file not found: {db_path}")
            print("Please ensure the database has been created first.")
            sys.exit(1)

    success = run_migration(db_path, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
