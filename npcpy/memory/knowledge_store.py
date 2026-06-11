import os
import uuid
import yaml
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any

DEFAULT_KNOWLEDGE_FILE = ".knowledge.yaml"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_id() -> str:
    return uuid.uuid4().hex


class KnowledgeStore:
    def __init__(self, directory: str):
        self.directory = os.path.abspath(directory)
        self.file_path = os.path.join(self.directory, DEFAULT_KNOWLEDGE_FILE)
        self._lock = threading.RLock()

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.file_path):
            return self._empty_template()
        with self._lock:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    return self._empty_template()
                data.setdefault("memories", [])
                data.setdefault("knowledge", [])
                return data
            except Exception:
                return self._empty_template()

    def save(self, data: Dict[str, Any]):
        data["updated_at"] = _utcnow()
        tmp = self.file_path + ".tmp"
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            os.replace(tmp, self.file_path)

    def _save_no_index(self, data: Dict[str, Any]):
        data["updated_at"] = _utcnow()
        tmp = self.file_path + ".tmp"
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            os.replace(tmp, self.file_path)

    def _empty_template(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "directory": self.directory,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "memories": [],
            "knowledge": [],
        }

    def append_memory(self,
                      mem_id: str = None,
                      message_id: str = "",
                      conversation_id: str = "",
                      npc: str = "",
                      team: str = "",
                      directory_path: str = "",
                      initial_memory: str = "",
                      status: str = "pending_approval",
                      model: str = "",
                      provider: str = "",
                      final_memory: str = None,
                      ) -> str:
        mem_id = mem_id or _make_id()
        mem = {
            "id": str(mem_id),
            "message_id": message_id,
            "conversation_id": conversation_id,
            "npc": npc,
            "team": team,
            "directory_path": directory_path,
            "timestamp": _utcnow(),
            "initial_memory": initial_memory,
            "final_memory": final_memory,
            "status": status,
            "model": model,
            "provider": provider,
            "created_at": _utcnow(),
        }
        data = self.load()
        data["memories"].append(mem)
        self.save(data)
        return mem_id

    def update_memory(self, mem_id: str, status: str, final_memory: str = None) -> bool:
        data = self.load()
        changed = False
        for mem in data.get("memories", []):
            if mem.get("id") == mem_id:
                mem["status"] = status
                if final_memory is not None:
                    mem["final_memory"] = final_memory
                changed = True
                break
        if changed:
            self.save(data)
        return changed

    def append_link(self, from_mem: str, to_mem: str, relation: str, agent: str = "") -> str:
        link = {
            "id": _make_id(),
            "from": from_mem,
            "to": to_mem,
            "relation": relation,
            "created_at": _utcnow(),
            "agent": agent,
        }
        data = self.load()
        data["knowledge"].append(link)
        self.save(data)
        return link["id"]

    def get_memories(self, status: str = None, limit: int = None) -> List[Dict[str, Any]]:
        data = self.load()
        mems = data.get("memories", [])
        if status:
            mems = [m for m in mems if m.get("status") == status]
        if limit:
            mems = mems[-limit:]
        return list(mems)

    def get_pending_memories(self) -> List[Dict[str, Any]]:
        return self.get_memories(status="pending_approval")

    def search_memories(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        q = query.lower()
        results = []
        for mem in self.get_memories():
            text = (mem.get("initial_memory") or mem.get("final_memory") or "").lower()
            if q in text:
                results.append(mem)
                if limit and len(results) >= limit:
                    break
        return results

    def get_links(self) -> List[Dict[str, Any]]:
        return list(self.load().get("knowledge", []))

    def get_links_for_memory(self, mem_id: str) -> Dict[str, List[Dict[str, Any]]]:
        data = self.load()
        out = []
        inn = []
        for link in data.get("knowledge", []):
            if link.get("from") == mem_id:
                out.append(link)
            if link.get("to") == mem_id:
                inn.append(link)
        return {"outgoing": out, "incoming": inn}

    def build_context(self, max_memories: int = 10) -> str:
        parts = []
        mems = self.get_memories(status="human-approved", limit=max_memories)
        if mems:
            parts.append("Local knowledge:")
            for m in mems:
                parts.append(f"- {m.get('final_memory') or m.get('initial_memory')}")
        return "\n".join(parts) if parts else ""

    @staticmethod
    def find_all(root_directory: str) -> List["KnowledgeStore"]:
        stores = []
        for dirpath, _dirnames, filenames in os.walk(root_directory):
            if DEFAULT_KNOWLEDGE_FILE in filenames:
                stores.append(KnowledgeStore(dirpath))
        return stores

    @staticmethod
    def aggregate(root_directory: str, max_depth: int = None) -> Dict[str, Any]:
        stores = KnowledgeStore.find_all(root_directory)
        if max_depth is not None:
            filtered = []
            for s in stores:
                rel = os.path.relpath(s.directory, root_directory).split(os.sep)
                if len(rel) <= max_depth:
                    filtered.append(s)
            stores = filtered
        all_memories = []
        all_knowledge = []
        for s in stores:
            d = s.load()
            all_memories.extend(d.get("memories", []))
            all_knowledge.extend(d.get("knowledge", []))
        return {
            "version": "1.0-aggregate",
            "root": root_directory,
            "sources": [s.directory for s in stores],
            "memories": all_memories,
            "knowledge": all_knowledge,
        }

    @staticmethod
    def init_directory(directory: str) -> "KnowledgeStore":
        store = KnowledgeStore(directory)
        if not os.path.exists(store.file_path):
            store.save(store._empty_template())
        return store


def get_store_for_path(path: str) -> KnowledgeStore:
    return KnowledgeStore.init_directory(path)
