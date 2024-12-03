# schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Coordinates(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    radius: Optional[float] = Field(1.0, gt=0.0)  # Radius in kilometers, default to 1 km

class AlertSummary(BaseModel):
    alert_id: int
    type_id: int
    type_name: str
    timesetamp: datetime

    class Config:
        from_attributes = True  # Updated from `orm_mode` to `from_attributes`

class InsightResponse(BaseModel):
    total_alerts: int
    alerts: List[AlertSummary]

class ErrorResponse(BaseModel):
    detail: str