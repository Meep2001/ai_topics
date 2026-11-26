from pydantic import BaseModel, Field
from typing import Literal

class SentimentAnalysis(BaseModel):
    sentiment: Literal[
        'positive',
        'negative',
        'neutral'
    ]
    reason: str
    confidence: float = Field(
        le=1,
        ge=0
    )