#!/usr/bin/env python3
"""
Migration: Move top-level resource directories into npc_team/.

Previously, directories like images/, models/, attachments/, mcp_servers/,
jobs/, triggers/, videos/, logs/ lived at the top level of the data dir
(e.g. ~/.npcsh/images/). They now belong inside npc_team/ (e.g.
~/.npcsh/npc_team/images/).

This migration moves contents from old locations to new ones, merging
if the destination already has files. Empty old directories are removed
after migration.

Usage:
    python migrate_dirs_to_npc_team.py [--data-dir PATH] [--dry-run]

Arguments:
    --data-dir: Path to the npcsh data directory (default: auto-detect via get_data_dir)
    --dry-run:  Show what would be done without making changes
"""

import os
import sys
import shutil
import argparse


DIRS_TO_MIGRATE = [
    "images",
    "models",
    "attachments",
    "mcp_servers",
    "jobs",
    "triggers",
    "videos",
    "logs",
]

MARKER_FILE = ".dirs_migrated_to_npc_team"


def _get_data_dir_fallback() -> str:
    """Resolve data dir without importing npcpy (for standalone use)."""
    npcsh_home = os.environ.get('INCOGNIDE_HOME')
    if npcsh_home:
        return os.path.expanduser(npcsh_home)

    import platform
    system = platform.system()
    if system == "Windows":
        base = os.environ.get('LOCALAPPDATA', os.path.expanduser('~/AppData/Local'))
        new_path = os.path.join(base, 'npcsh')
    elif system == "Darwin":
        new_path = os.path.expanduser('~/Library/Application Support/npcsh')
    else:
        xdg_data = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))
        new_path = os.path.join(xdg_data, 'npcsh')

    old_path = os.path.expanduser('~/.npcsh')
    if os.path.exists(old_path) and not os.path.exists(new_path):
        return old_path

    return new_path


def get_data_dir() -> str:
    """Try importing from npcpy first, fall back to local logic."""
    try:
        from npcpy.npc_sysenv import get_data_dir as _gdd
        return _gdd()
    except ImportError:
        return _get_data_dir_fallback()


def is_migrated(data_dir: str) -> bool:
    """Check if migration has already been applied."""
    return os.path.isfile(os.path.join(data_dir, MARKER_FILE))


def _move_contents(src: str, dst: str, dry_run: bool = False) -> tuple:
    """Move all items from src into dst, merging with existing content.

    Returns (moved_count, skipped_count).
    """
    moved = 0
    skipped = 0
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)

        if os.path.exists(dst_path):
            # If both are directories, recurse to merge
            if os.path.isdir(src_path) and os.path.isdir(dst_path):
                print(f"    MERGE: {item}/ (both dirs exist, recursing)")
                if not dry_run:
                    sub_m, sub_s = _move_contents(src_path, dst_path, dry_run)
                    moved += sub_m
                    skipped += sub_s
                    # Remove src dir if now empty
                    if not os.listdir(src_path):
                        os.rmdir(src_path)
                else:
                    sub_m, sub_s = _move_contents(src_path, dst_path, dry_run)
                    moved += sub_m
                    skipped += sub_s
            else:
                print(f"    SKIP:  {item} (already exists at destination)")
                skipped += 1
        else:
            print(f"    MOVE:  {item}")
            if not dry_run:
                shutil.move(src_path, dst_path)
            moved += 1

    return moved, skipped


def run_migration(data_dir: str, dry_run: bool = False) -> bool:
    """Run the directory migration.

    Returns True if migration was applied (or already done), False on error.
    """
    print(f"Migration: Move resource directories into npc_team/")
    print(f"Data dir:  {data_dir}")
    print(f"Dry run:   {dry_run}")
    print("-" * 60)

    if not os.path.isdir(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        print("Nothing to migrate.")
        return True

    if is_migrated(data_dir):
        print("Already migrated (marker file exists). Skipping.")
        return True

    npc_team_dir = os.path.join(data_dir, "npc_team")
    os.makedirs(npc_team_dir, exist_ok=True)

    total_moved = 0
    total_skipped = 0
    dirs_processed = 0

    for dirname in DIRS_TO_MIGRATE:
        old_dir = os.path.join(data_dir, dirname)
        new_dir = os.path.join(npc_team_dir, dirname)

        if not os.path.isdir(old_dir):
            continue

        # Check if old dir is empty
        if not os.listdir(old_dir):
            print(f"\n  [{dirname}/] Empty, removing.")
            if not dry_run:
                os.rmdir(old_dir)
            continue

        print(f"\n  [{dirname}/] -> npc_team/{dirname}/")
        os.makedirs(new_dir, exist_ok=True)

        moved, skipped = _move_contents(old_dir, new_dir, dry_run)
        total_moved += moved
        total_skipped += skipped
        dirs_processed += 1

        # Remove old dir if now empty
        if not dry_run and os.path.isdir(old_dir) and not os.listdir(old_dir):
            os.rmdir(old_dir)
            print(f"    Removed empty {dirname}/")

    # Write marker file
    if not dry_run:
        with open(os.path.join(data_dir, MARKER_FILE), "w") as f:
            f.write("Migration applied: top-level dirs moved into npc_team/\n")

    print("\n" + "=" * 60)
    if dry_run:
        print(f"[DRY RUN] Would move {total_moved} items, skip {total_skipped}")
    else:
        print("Migration completed!")
        print(f"  Items moved:   {total_moved}")
        print(f"  Items skipped: {total_skipped}")
        print(f"  Dirs processed: {dirs_processed}")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Move top-level resource directories into npc_team/'
    )
    parser.add_argument(
        '--data-dir',
        default=None,
        help='Path to the npcsh data directory (default: auto-detect)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    data_dir = args.data_dir or get_data_dir()
    data_dir = os.path.expanduser(data_dir)

    success = run_migration(data_dir, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
