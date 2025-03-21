# models.py

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime

Base = declarative_base()

class Alert(Base):
    __tablename__ = 'alerts'
    __table_args__ = {'schema': 'map'}

    id = Column(Integer, primary_key=True, index=True)
    device_type = Column(String)
    device_type_id = Column(Integer)
    date_created = Column(DateTime, default=datetime.utcnow)
    sensor_type = Column(String)
    event_type = Column(String)
    event_type_id = Column(Integer)
    resolution_reason = Column(String)
    industry = Column(String)
    industry_id = Column(Integer)
    longitude = Column(Float)
    latitude = Column(Float)
    country = Column(String)
    resolution_reason_id = Column(Integer)
    sensor_type_id = Column(Integer)
    country_id = Column(Integer)