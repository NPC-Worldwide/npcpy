#!/usr/bin/env python3
"""
Migration script to add device_id and device_name columns to relevant tables.

This allows tracking which machine/device created each record for multi-device sync.

Usage:
    python add_device_id_column.py [--db-path PATH]

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
    """Run the migration to add device columns to all relevant tables"""
    print(f"Migration: Add device_id and device_name columns")
    print(f"Database: {db_path}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)

    engine = get_engine(db_path)

    # Tables that need device tracking
    tables_to_update = [
        'conversation_history',
        'command_history',
        'memory_lifecycle',
        'jinx_executions',
        'npc_executions',
        'labels',
    ]

    # Columns to add to each table
    new_columns = [
        ('device_id', 'VARCHAR(255)'),    # UUID of the device
        ('device_name', 'VARCHAR(255)'),  # Human-readable device name
    ]

    total_added = 0
    total_skipped = 0

    for table_name in tables_to_update:
        if not table_exists(engine, table_name):
            print(f"\n[{table_name}] Table does not exist, skipping")
            continue

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
        description='Add device_id and device_name columns to tables for multi-device sync'
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
