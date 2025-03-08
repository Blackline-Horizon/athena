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
def get_report_data(resolution:GetReport):
    data = {
        'time_series_overall': {
            'date_created': ['2023-01', '2023-02', '2023-03', '2023-04'],
            'alert_count': [100, 150, 120, 180]
        },
        'alerts_last_4w': 200,
        'last_4w': 170,
        'predicted_last_4w': 190,
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

@app.get("/events")
def get_events():
    try:
        # Path to the map-data.json file
        json_file_path = os.path.join(os.getcwd(), "Mock Data", "map-data.json")
        
        # Read the JSON file
        with open(json_file_path, "r") as file:
            events_data = json.load(file)

        return {"events": events_data["events"]}  # Return the events part of the JSON

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insights/alerts", response_model=InsightResponse, responses={500: {"model": ErrorResponse}})
def get_alerts_nearby(coordinates: Coordinates):
    db = SessionLocal()
    try:
        # Build a point from input coordinates
        point_wkt = f'POINT({coordinates.longitude} {coordinates.latitude})'
        point = func.ST_GeogFromText(point_wkt)

        # Define the radius in meters
        radius_meters = coordinates.radius * 1000  # Convert km to meters

        # Perform spatial query to find alerts within the radius
        alerts = db.query(Alert, Type).join(Type).filter(
            func.ST_DWithin(
                Alert.coordinates,
                point,
                radius_meters
            )
        ).all()

        # Aggregate results
        alert_summaries = [
            AlertSummary(
                alert_id=alert.Alert.alert_id,
                type_id=alert.Alert.type_id,
                type_name=alert.Type.name,
                timesetamp=alert.Alert.timesetamp
            )
            for alert in alerts
        ]

        response = InsightResponse(
            total_alerts=len(alert_summaries),
            alerts=alert_summaries
        )

        return response
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")