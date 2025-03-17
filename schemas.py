# schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class AlertOut(BaseModel):
    id: int
    device_type: Optional[str] = None
    date_created: datetime
    sensor_type: Optional[str] = None
    event_type: Optional[str] = None
    resolution_reason: Optional[str] = None
    industry: Optional[str] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    country: Optional[str] = None

    class Config:
        orm_mode = True

class GetDashboard(BaseModel):
    sensor_type: Optional[List[str]] = None
    industry: Optional[List[str]] = None
    event_type: Optional[List[str]] = None
    resolution_reason: Optional[List[str]] = None
    device_type: Optional[List[str]] = None
    country: Optional[List[str]] = None
    continent: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class GetReport(BaseModel):
    resolution_reason: Optional[List[str]] = None
    device_type: Optional[List[str]] = None
    sensor_type: Optional[List[str]] = None
    event_type: Optional[List[str]] = None
    industry: Optional[List[str]] = None
    continent: Optional[List[str]] = None
    date_start: datetime = Field(...)
    date_end: datetime = Field(...)
    country: Optional[List[str]] = None

class ErrorResponse(BaseModel):
    detail: str