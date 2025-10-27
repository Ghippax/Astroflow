from pydantic import BaseModel
from typing import Optional

"""Validation logic and structure for plotting parameters
"""

class PropParams(BaseModel):
    save: Optional[str] = None