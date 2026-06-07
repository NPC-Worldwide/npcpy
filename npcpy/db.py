import json
import os
import uuid
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def generate_message_id() -> str:
    return str(uuid.uuid4())


def normalize_path_for_db(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return path_str
    return path_str.replace("\\", "/").rstrip("/")


def deep_to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: deep_to_dict(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [deep_to_dict(item) for item in obj]
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict", None)):
        return deep_to_dict(obj.to_dict())
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return None


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return deep_to_dict(obj)
        except TypeError:
            return super().default(obj)


def ensure_engine(conn: Union[str, Engine]) -> Engine:
    if isinstance(conn, Engine):
        return conn
    if conn.startswith("~/"):
        conn = os.path.expanduser(conn)
    return create_engine(conn)
