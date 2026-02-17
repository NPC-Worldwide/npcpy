from pydantic import BaseModel
from typing import List, Dict

class NPC_Model(BaseModel):
    name: str
    primary_directive: str
    model: str
    provider: str
    api_url: str
    jinxs: List[str]

class Jinx_Model(BaseModel):
    jinx_name: str
    description: str
    steps: List[Dict[str, str]]

class JinxStep_Model(BaseModel):
    engine: str
    code: str

class Context_Model(BaseModel):
    databases: List[str]
    files: List[str]
    vars: List[Dict[str, str]]

class Pipeline_Model(BaseModel):
    steps: List[Dict[str, str]]

class PipelineStep_Model(BaseModel):
    jinx: str
    args: List[str]
    model: str
    provider: str
    task: str
    npc: str

