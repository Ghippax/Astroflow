from pydantic import BaseModel
from typing import Optional

"""Validation logic and structure for analysis parameters
"""

class PropParams(BaseModel):
    save: Optional[str] = None