# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import json
from dotenv import load_dotenv
import os

from models import Base, Type, Alert
from schemas import Coordinates, AlertSummary, InsightResponse, ErrorResponse, GetReport

# Load environment variables
load_dotenv(dotenv_path=".env")

# Get values from environment variables
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 3001))
DATABASE_URL = os.getenv("DATABASE_URL")


app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create synchronous engine
engine = create_engine(
    DATABASE_URL,
    echo=True,  # Set to False in production
)

# Create sessionmaker with autoflush enabled
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=True,
    bind=engine
)

@app.on_event("startup")
def startup_event():
    # Create the 'map' schema if it doesn't exist
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS map"))
        conn.commit()
    # Create tables within the 'map' schema
    Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Alert Insights Service is up and running!"}

@app.get("/report_data")
def get_report_data(filters:GetReport):
    data = {
        'time_series_overall': {
            'date_created': ['2023-01', '2023-02', '2023-03', '2023-04'],
            'alert_count': [100, 150, 120, 180]
        },
        # Grouped data for various categories, with groups indexed by integers
        'grouped_data': {
            "resolution_reason": {
                '2023-01': {1: 30, -1: 70},
                '2023-02': {1: 50, -1: 100},
                '2023-03': {1: 40, -1: 80},
                '2023-04': {1: 60, -1: 120}
            },
            "device_type": {
                # Add similar structure here when data is available
            },
            "sensor_type": {
                # Add similar structure here when data is available
            },
            "industry": {
                # Add similar structure here when data is available
            },
            "event_type": {
                # Add similar structure here when data is available
            }
        }
    }

    return data



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")