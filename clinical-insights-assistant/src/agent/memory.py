from pydantic import BaseModel
from typing import List, Optional

class AgentState(BaseModel):
    query: str
    selected_tool: Optional[str] = None
    result_text: Optional[str] = None
    notes_sample: Optional[List[str]] = None
    summary: Optional[str] = None
    delta_mg: Optional[float] = None
