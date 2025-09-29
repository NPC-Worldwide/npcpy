from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading
import queue
import time

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

def memory_approval_ui(memories: List[Dict]) -> List[Dict]:
    if not memories:
        return []
    
    print(f"\n📝 {len(memories)} memories ready for approval:")
    
    approvals = []
    for i, memory in enumerate(memories, 1):
        print(f"\n--- Memory {i}/{len(memories)} ---")
        print(f"NPC: {memory['npc']}")
        content_preview = memory['content'][:200]
        if len(memory['content']) > 200:
            content_preview += '...'
        print(f"Content: {content_preview}")
        
        while True:
            choice = input(
                "(a)pprove, (r)eject, (e)dit, (s)kip | "
                "(A)ll approve, (R)all reject, (S)all skip: "
            ).strip().lower()
            
            if choice == 'a':
                approvals.append({
                    "memory_id": memory['memory_id'],
                    "decision": "human-approved"
                })
                break
            elif choice == 'r':
                approvals.append({
                    "memory_id": memory['memory_id'],
                    "decision": "human-rejected"
                })
                break
            elif choice == 'e':
                edited = input("Edit memory: ").strip()
                if edited:
                    approvals.append({
                        "memory_id": memory['memory_id'],
                        "decision": "human-edited",
                        "final_memory": edited
                    })
                break
            elif choice == 's':
                break
            elif choice == 'A':
                for remaining_memory in memories[i-1:]:
                    approvals.append({
                        "memory_id": remaining_memory['memory_id'],
                        "decision": "human-approved"
                    })
                return approvals
            elif choice == 'R':
                for remaining_memory in memories[i-1:]:
                    approvals.append({
                        "memory_id": remaining_memory['memory_id'],
                        "decision": "human-rejected"
                    })
                return approvals
            elif choice == 'S':
                return approvals
    
    return approvals