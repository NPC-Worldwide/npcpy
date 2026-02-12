#!/usr/bin/env python3
"""
Migration script to clean up KG tables:
  1. Remove duplicate facts and concepts
  2. Rebuild kg_facts and kg_concepts with UNIQUE constraints
  3. Deduplicate kg_metadata (keep latest generation per scope)
  4. Backfill NULL type/source_text columns

Usage:
    python cleanup_kg_tables.py [--db-path PATH] [--dry-run]

Arguments:
    --db-path: Path to the SQLite database (default: ~/npcsh_history.db)
    --dry-run: Show what would be done without making changes
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
    """Create SQLAlchemy engine from database path."""
    if db_path.startswith('~/'):
        db_path = os.path.expanduser(db_path)

    if db_path.startswith('postgresql://') or db_path.startswith('postgres://'):
        return create_engine(db_path)
    else:
        return create_engine(f'sqlite:///{db_path}')


def table_exists(engine, table_name: str) -> bool:
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def has_unique_constraint(engine, table_name: str, columns: list) -> bool:
    """Check if the table already has a UNIQUE constraint on the given columns."""
    inspector = inspect(engine)
    for uc in inspector.get_unique_constraints(table_name):
        if set(uc['column_names']) == set(columns):
            return True
    # Also check indexes for uniqueness
    for idx in inspector.get_indexes(table_name):
        if idx.get('unique') and set(idx['column_names']) == set(columns):
            return True
    return False


def run_migration(db_path: str, dry_run: bool = False):
    """Run the KG cleanup migration."""
    print("Migration: Clean up KG tables (dedup + UNIQUE constraints)")
    print(f"Database: {db_path}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)

    engine = get_engine(db_path)
    is_sqlite = 'sqlite' in str(engine.url)

    if not table_exists(engine, 'kg_facts'):
        print("\n[kg_facts] Table does not exist. Nothing to migrate.")
        return True

    # ---------------------------------------------------------------
    # Step 1: Report duplicates
    # ---------------------------------------------------------------
    print("\n[Step 1] Scanning for duplicates...")

    with engine.connect() as conn:
        dup_facts = conn.execute(text("""
            SELECT statement, team_name, npc_name, directory_path, COUNT(*) as cnt
            FROM kg_facts
            GROUP BY statement, team_name, npc_name, directory_path
            HAVING COUNT(*) > 1
        """)).fetchall()

        dup_concepts = conn.execute(text("""
            SELECT name, team_name, npc_name, directory_path, COUNT(*) as cnt
            FROM kg_concepts
            GROUP BY name, team_name, npc_name, directory_path
            HAVING COUNT(*) > 1
        """)).fetchall()

        dup_meta = conn.execute(text("""
            SELECT key, team_name, npc_name, directory_path, COUNT(*) as cnt
            FROM kg_metadata
            GROUP BY key, team_name, npc_name, directory_path
            HAVING COUNT(*) > 1
        """)).fetchall()

        total_facts = conn.execute(text("SELECT COUNT(*) FROM kg_facts")).scalar()
        total_concepts = conn.execute(text("SELECT COUNT(*) FROM kg_concepts")).scalar()

    print(f"  Facts:    {total_facts} total, {len(dup_facts)} groups with duplicates")
    for row in dup_facts:
        print(f"    - '{row[0][:60]}...' ({row[1]}/{row[2]}) x{row[4]}")

    print(f"  Concepts: {total_concepts} total, {len(dup_concepts)} groups with duplicates")
    for row in dup_concepts:
        print(f"    - '{row[0]}' ({row[1]}/{row[2]}) x{row[4]}")

    print(f"  Metadata: {len(dup_meta)} groups with duplicates")
    for row in dup_meta:
        print(f"    - key='{row[0]}' ({row[1]}/{row[2]}) x{row[4]}")

    if dry_run:
        print("\n[DRY RUN] Would perform the following:")
        print("  - Deduplicate kg_facts, kg_concepts, kg_metadata")
        print("  - Rebuild tables with UNIQUE constraints")
        print("  - Backfill NULL type → 'organic', NULL source_text → ''")
        print("=" * 60)
        return True

    # ---------------------------------------------------------------
    # Step 2: Rebuild kg_facts with UNIQUE constraint
    # ---------------------------------------------------------------
    if is_sqlite and not has_unique_constraint(engine, 'kg_facts',
            ['statement', 'team_name', 'npc_name', 'directory_path']):
        print("\n[Step 2] Rebuilding kg_facts with UNIQUE constraint...")

        with engine.begin() as conn:
            # Create new table with UNIQUE constraint
            conn.execute(text("""
                CREATE TABLE kg_facts_new (
                    statement TEXT NOT NULL,
                    team_name VARCHAR(255) NOT NULL,
                    npc_name VARCHAR(255) NOT NULL,
                    directory_path TEXT NOT NULL,
                    source_text TEXT,
                    type VARCHAR(100),
                    generation INTEGER,
                    origin VARCHAR(100),
                    UNIQUE(statement, team_name, npc_name, directory_path)
                )
            """))

            # Copy deduplicated data (keep the row with highest generation)
            conn.execute(text("""
                INSERT INTO kg_facts_new
                    (statement, team_name, npc_name, directory_path,
                     source_text, type, generation, origin)
                SELECT statement, team_name, npc_name, directory_path,
                       COALESCE(source_text, ''),
                       COALESCE(type, 'organic'),
                       MAX(generation),
                       origin
                FROM kg_facts
                GROUP BY statement, team_name, npc_name, directory_path
            """))

            new_count = conn.execute(text("SELECT COUNT(*) FROM kg_facts_new")).scalar()
            print(f"  Copied {new_count} unique facts (was {total_facts})")

            conn.execute(text("DROP TABLE kg_facts"))
            conn.execute(text("ALTER TABLE kg_facts_new RENAME TO kg_facts"))

        print("  SUCCESS: kg_facts rebuilt with UNIQUE constraint")
    else:
        print("\n[Step 2] SKIP: kg_facts already has UNIQUE constraint or not SQLite")

    # ---------------------------------------------------------------
    # Step 3: Rebuild kg_concepts with UNIQUE constraint
    # ---------------------------------------------------------------
    if is_sqlite and not has_unique_constraint(engine, 'kg_concepts',
            ['name', 'team_name', 'npc_name', 'directory_path']):
        print("\n[Step 3] Rebuilding kg_concepts with UNIQUE constraint...")

        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE kg_concepts_new (
                    name TEXT NOT NULL,
                    team_name VARCHAR(255) NOT NULL,
                    npc_name VARCHAR(255) NOT NULL,
                    directory_path TEXT NOT NULL,
                    generation INTEGER,
                    origin VARCHAR(100),
                    UNIQUE(name, team_name, npc_name, directory_path)
                )
            """))

            conn.execute(text("""
                INSERT INTO kg_concepts_new
                    (name, team_name, npc_name, directory_path, generation, origin)
                SELECT name, team_name, npc_name, directory_path,
                       MAX(generation), origin
                FROM kg_concepts
                GROUP BY name, team_name, npc_name, directory_path
            """))

            new_count = conn.execute(text("SELECT COUNT(*) FROM kg_concepts_new")).scalar()
            print(f"  Copied {new_count} unique concepts (was {total_concepts})")

            conn.execute(text("DROP TABLE kg_concepts"))
            conn.execute(text("ALTER TABLE kg_concepts_new RENAME TO kg_concepts"))

        print("  SUCCESS: kg_concepts rebuilt with UNIQUE constraint")
    else:
        print("\n[Step 3] SKIP: kg_concepts already has UNIQUE constraint or not SQLite")

    # ---------------------------------------------------------------
    # Step 4: Deduplicate kg_metadata
    # ---------------------------------------------------------------
    if dup_meta:
        print("\n[Step 4] Deduplicating kg_metadata...")

        with engine.begin() as conn:
            # For each duplicate group, keep the one with the highest value
            # (generation is stored as string, so cast for comparison)
            for row in dup_meta:
                key, team, npc, dirpath, cnt = row
                # Delete all but keep one with max value
                conn.execute(text("""
                    DELETE FROM kg_metadata
                    WHERE rowid NOT IN (
                        SELECT rowid FROM kg_metadata
                        WHERE key = :key AND team_name = :team
                        AND npc_name = :npc AND directory_path = :dirpath
                        ORDER BY CAST(value AS INTEGER) DESC
                        LIMIT 1
                    )
                    AND key = :key AND team_name = :team
                    AND npc_name = :npc AND directory_path = :dirpath
                """), {"key": key, "team": team, "npc": npc, "dirpath": dirpath})

                print(f"  Deduped: key='{key}' for {team}/{npc} (removed {cnt - 1} extra rows)")
    else:
        print("\n[Step 4] SKIP: No duplicate metadata entries")

    # ---------------------------------------------------------------
    # Step 5: Final report
    # ---------------------------------------------------------------
    print("\n[Step 5] Final state:")

    with engine.connect() as conn:
        final_facts = conn.execute(text("SELECT COUNT(*) FROM kg_facts")).scalar()
        final_concepts = conn.execute(text("SELECT COUNT(*) FROM kg_concepts")).scalar()
        final_links = conn.execute(text("SELECT COUNT(*) FROM kg_links")).scalar()
        final_meta = conn.execute(text("SELECT COUNT(*) FROM kg_metadata")).scalar()

    print(f"  kg_facts:    {final_facts}")
    print(f"  kg_concepts: {final_concepts}")
    print(f"  kg_links:    {final_links}")
    print(f"  kg_metadata: {final_meta}")

    print("\n" + "=" * 60)
    print("Migration completed!")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Clean up KG tables: deduplicate and add UNIQUE constraints'
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
    db_path = os.path.expanduser(args.db_path)

    if not db_path.startswith('postgresql://') and not db_path.startswith('postgres://'):
        if not os.path.exists(db_path):
            print(f"ERROR: Database file not found: {db_path}")
            sys.exit(1)

    success = run_migration(db_path, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
