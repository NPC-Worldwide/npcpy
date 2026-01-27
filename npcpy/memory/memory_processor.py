from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading
import queue
import time
import os

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, on_color=None, attrs=None):
        return text

@dataclass
class MemoryItem:
    message_id: str
    conversation_id: str
    npc: str
    team: str
    directory_path: str
    content: str
    context: str
    model: str
    provider: str


def _clear_line():
    """Clear current line in terminal."""
    print('\r' + ' ' * 80 + '\r', end='')


def _print_header(title: str, width: int = 60):
    """Print a styled header."""
    print(colored("=" * width, "cyan"))
    print(colored(f"  {title}", "cyan", attrs=["bold"]))
    print(colored("=" * width, "cyan"))


def _print_memory_box(memory: Dict, index: int, total: int):
    """Print a memory in a nice box format."""
    width = 70

    # Header with progress
    progress = f"[{index}/{total}]"
    npc_info = f"NPC: {memory.get('npc', 'unknown')}"
    header = f"{progress} {npc_info}"
    print(colored("+" + "-" * (width - 2) + "+", "blue"))
    print(colored(f"| {header:<{width-4}} |", "blue"))
    print(colored("+" + "-" * (width - 2) + "+", "blue"))

    # Content
    content = memory.get('content', '')
    # Wrap content to fit in box
    lines = []
    words = content.split()
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= width - 6:
            current_line += (" " if current_line else "") + word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    for line in lines[:6]:  # Max 6 lines
        print(colored(f"|  {line:<{width-5}} |", "white"))

    if len(lines) > 6:
        print(colored(f"|  {'...':<{width-5}} |", "grey"))

    # Context if available
    ctx = memory.get('context', '')
    if ctx:
        print(colored("+" + "-" * (width - 2) + "+", "blue"))
        ctx_short = ctx[:width-8] + "..." if len(ctx) > width - 8 else ctx
        print(colored(f"| {ctx_short:<{width-4}} |", "grey"))

    print(colored("+" + "-" * (width - 2) + "+", "blue"))


def _print_options():
    """Print available options."""
    print()
    options = [
        (colored("a", "green", attrs=["bold"]), "approve"),
        (colored("r", "red", attrs=["bold"]), "reject"),
        (colored("e", "yellow", attrs=["bold"]), "edit"),
        (colored("s", "grey"), "skip"),
        (colored("A", "green"), "approve all"),
        (colored("R", "red"), "reject all"),
        (colored("D", "cyan"), "defer (review later)"),
    ]
    print("  " + "  |  ".join([f"({k}) {v}" for k, v in options]))


def _print_summary(stats: Dict):
    """Print approval summary."""
    print()
    _print_header("Memory Review Summary")
    print(f"  {colored('Approved:', 'green')} {stats.get('approved', 0)}")
    print(f"  {colored('Rejected:', 'red')} {stats.get('rejected', 0)}")
    print(f"  {colored('Edited:', 'yellow')} {stats.get('edited', 0)}")
    print(f"  {colored('Skipped:', 'grey')} {stats.get('skipped', 0)}")
    print(f"  {colored('Deferred:', 'cyan')} {stats.get('deferred', 0)}")
    print()


def memory_approval_ui(memories: List[Dict], show_context: bool = True) -> List[Dict]:
    """
    Enhanced memory approval UI with better formatting.

    Args:
        memories: List of memory dicts with 'memory_id', 'content', 'npc', 'context'
        show_context: Whether to show context info

    Returns:
        List of approval dicts with 'memory_id', 'decision', optionally 'final_memory'
    """
    if not memories:
        return []

    # Stats tracking
    stats = {'approved': 0, 'rejected': 0, 'edited': 0, 'skipped': 0, 'deferred': 0}
    approvals = []

    print()
    _print_header(f"Memory Review - {len(memories)} memories")
    print()

    i = 0
    while i < len(memories):
        memory = memories[i]
        os.system('clear' if os.name == 'posix' else 'cls') if len(memories) > 3 else None

        _print_memory_box(memory, i + 1, len(memories))
        _print_options()

        try:
            choice = input("\n  Your choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Review cancelled.")
            break

        if choice == 'a':
            approvals.append({
                "memory_id": memory['memory_id'],
                "decision": "human-approved"
            })
            stats['approved'] += 1
            print(colored("  ✓ Approved", "green"))
            i += 1

        elif choice == 'r':
            approvals.append({
                "memory_id": memory['memory_id'],
                "decision": "human-rejected"
            })
            stats['rejected'] += 1
            print(colored("  ✗ Rejected", "red"))
            i += 1

        elif choice == 'e':
            print(colored("\n  Current:", "grey"), memory['content'][:100])
            print(colored("  Enter new text (or empty to cancel):", "yellow"))
            try:
                edited = input("  > ").strip()
                if edited:
                    approvals.append({
                        "memory_id": memory['memory_id'],
                        "decision": "human-edited",
                        "final_memory": edited
                    })
                    stats['edited'] += 1
                    print(colored("  ✎ Edited and approved", "yellow"))
                    i += 1
                else:
                    print(colored("  Edit cancelled", "grey"))
            except (EOFError, KeyboardInterrupt):
                print(colored("  Edit cancelled", "grey"))

        elif choice == 's':
            stats['skipped'] += 1
            print(colored("  ○ Skipped", "grey"))
            i += 1

        elif choice == 'A':
            # Approve all remaining
            for remaining in memories[i:]:
                approvals.append({
                    "memory_id": remaining['memory_id'],
                    "decision": "human-approved"
                })
                stats['approved'] += 1
            print(colored(f"  ✓ Approved all {len(memories) - i} remaining", "green"))
            break

        elif choice == 'R':
            # Reject all remaining
            for remaining in memories[i:]:
                approvals.append({
                    "memory_id": remaining['memory_id'],
                    "decision": "human-rejected"
                })
                stats['rejected'] += 1
            print(colored(f"  ✗ Rejected all {len(memories) - i} remaining", "red"))
            break

        elif choice == 'D':
            # Defer - don't add to approvals, will remain pending
            stats['deferred'] += len(memories) - i
            print(colored(f"  ⏸ Deferred {len(memories) - i} memories for later review", "cyan"))
            break

        elif choice == 'q':
            print(colored("  Review ended", "grey"))
            break

        else:
            print(colored("  Invalid choice. Use a/r/e/s/A/R/D", "red"))

        time.sleep(0.2)  # Brief pause for readability

    _print_summary(stats)
    return approvals


def memory_batch_review_ui(
    command_history,
    npc_filter: str = None,
    team_filter: str = None,
    limit: int = 50
) -> Dict[str, int]:
    """
    Review pending memories from the database in batch.

    Args:
        command_history: CommandHistory instance
        npc_filter: Optional NPC name filter
        team_filter: Optional team name filter
        limit: Max memories to review

    Returns:
        Dict with counts of approved/rejected/etc
    """
    # Get pending memories
    pending = command_history.get_pending_memories(limit=limit)

    if not pending:
        print(colored("No pending memories to review.", "grey"))
        return {'approved': 0, 'rejected': 0, 'edited': 0, 'skipped': 0}

    # Filter if specified
    if npc_filter:
        pending = [m for m in pending if m.get('npc') == npc_filter]
    if team_filter:
        pending = [m for m in pending if m.get('team') == team_filter]

    if not pending:
        print(colored("No memories match the filter criteria.", "grey"))
        return {'approved': 0, 'rejected': 0, 'edited': 0, 'skipped': 0}

    # Convert to format expected by approval UI
    memories_for_ui = []
    for m in pending:
        memories_for_ui.append({
            'memory_id': m.get('id'),
            'content': m.get('initial_memory', ''),
            'npc': m.get('npc', 'unknown'),
            'context': f"Team: {m.get('team', 'unknown')} | Path: {m.get('directory_path', '')[:30]}"
        })

    # Run approval UI
    approvals = memory_approval_ui(memories_for_ui)

    # Apply approvals to database
    stats = {'approved': 0, 'rejected': 0, 'edited': 0, 'skipped': 0}

    for approval in approvals:
        memory_id = approval['memory_id']
        decision = approval['decision']
        final_memory = approval.get('final_memory')

        command_history.update_memory_status(memory_id, decision, final_memory)

        if 'approved' in decision:
            stats['approved'] += 1
        elif 'rejected' in decision:
            stats['rejected'] += 1
        elif 'edited' in decision:
            stats['edited'] += 1

    stats['skipped'] = len(pending) - len(approvals)

    return stats