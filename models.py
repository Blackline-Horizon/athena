# models.py

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, MetaData
from sqlalchemy.orm import relationship
from geoalchemy2 import Geography
from datetime import datetime

# Specify the schema
metadata = MetaData(schema='map')

Base = declarative_base(metadata=metadata)

class Type(Base):
    __tablename__ = 'types'

    type_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)

    alerts = relationship('Alert', back_populates='type')

class Alert(Base):
    __tablename__ = 'alerts'

    alert_id = Column(Integer, primary_key=True, index=True)
    type_id = Column(Integer, ForeignKey('map.types.type_id'), nullable=False)
    timesetamp = Column(DateTime, default=datetime.utcnow)
    coordinates = Column(Geography(geometry_type='POINT', srid=4326), nullable=False)

    type = relationship('Type', back_populates='alerts')