#!/usr/bin/env python3
"""
Migration script to add input_tokens, output_tokens, and cost columns to conversation_history.

This allows tracking token usage and API costs per message.

Usage:
    python add_token_cost_columns.py [--db-path PATH]

Arguments:
    --db-path: Path to the SQLite database (default: ~/npcsh_history.db)
"""

import os
import sys
import argparse

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
    try:
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except Exception:
        return False


def table_exists(engine, table_name: str) -> bool:
    """Check if a table exists"""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def run_migration(db_path: str, dry_run: bool = False):
    """Run the migration to add token and cost columns"""
    print(f"Migration: Add input_tokens, output_tokens, and cost columns")
    print(f"Database: {db_path}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)

    engine = get_engine(db_path)

    table_name = 'conversation_history'

    # Columns to add
    new_columns = [
        ('input_tokens', 'INTEGER'),      # Input/prompt token count
        ('output_tokens', 'INTEGER'),     # Output/completion token count
        ('cost', 'VARCHAR(50)'),          # Cost in USD (string for precision)
    ]

    total_added = 0
    total_skipped = 0

    if not table_exists(engine, table_name):
        print(f"\n[{table_name}] Table does not exist!")
        return False

    print(f"\n[{table_name}]")

    for column_name, column_type in new_columns:
        if column_exists(engine, table_name, column_name):
            print(f"  SKIP: Column '{column_name}' already exists")
            total_skipped += 1
        else:
            print(f"  ADD:  Column '{column_name}' ({column_type})")

            if not dry_run:
                try:
                    with engine.begin() as conn:
                        sql = text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                        conn.execute(sql)
                    print(f"  SUCCESS: Added column '{column_name}'")
                    total_added += 1
                except Exception as e:
                    print(f"  ERROR: Failed to add column '{column_name}': {e}")
            else:
                total_added += 1

    print("\n" + "=" * 60)
    if dry_run:
        print(f"[DRY RUN] Would add {total_added} columns")
    else:
        print("Migration completed!")
        print(f"  Columns added: {total_added}")
    print(f"  Columns skipped (already exist): {total_skipped}")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Add input_tokens, output_tokens, and cost columns for token tracking'
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
