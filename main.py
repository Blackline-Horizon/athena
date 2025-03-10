from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import json
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import List, Optional

# Load environment variables
load_dotenv(dotenv_path=".env")

# Get values from environment variables
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 3001))

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Alert Insights Service is up and running!"}

@app.get("/alerts")
def get_alerts(
    sensor_types: Optional[List[str]] = Query(None),
    industries: Optional[List[str]] = Query(None),
    event_types: Optional[List[str]] = Query(None),
    resolution_reasons: Optional[List[str]] = Query(None),
    device_types: Optional[List[str]] = Query(None),
    countries: Optional[List[str]] = Query(None),
    continents: Optional[List[str]] = Query(None),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get filtered alerts based on provided parameters
    """
    try:
        # Path to the alerts.json file
        json_file_path = os.path.join(os.getcwd(), "Mock Data", "alerts.json")
        
        # Read the JSON file
        with open(json_file_path, "r") as file:
            data = json.load(file)
        
        # Get all alerts
        alerts = data["alerts"]
        meta = data["meta"]
        
        # Apply filters
        filtered_alerts = alerts
        
        # Filter by sensor type
        if sensor_types:
            filtered_alerts = [a for a in filtered_alerts if a["sensor_type"] in sensor_types]
            
        # Filter by industry
        if industries:
            filtered_alerts = [a for a in filtered_alerts if a["industry"] in industries]
            
        # Filter by event type
        if event_types:
            filtered_alerts = [a for a in filtered_alerts if a["event_type"] in event_types]
            
        # Filter by resolution reason
        if resolution_reasons:
            filtered_alerts = [a for a in filtered_alerts if a["resolution_reason"] in resolution_reasons]
            
        # Filter by device type
        if device_types:
            filtered_alerts = [a for a in filtered_alerts if a["device_type"] in device_types]
            
        # Filter by country
        if countries:
            filtered_alerts = [a for a in filtered_alerts if a["country_name"] in countries]
            
        # Filter by continent
        if continents:
            continent_countries = []
            for continent in continents:
                if continent in meta["countries"]:
                    continent_countries.extend(meta["countries"][continent])
            
            filtered_alerts = [a for a in filtered_alerts if a["country_name"] in continent_countries]
            
        # Filter by date range
        if start_date:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            filtered_alerts = [a for a in filtered_alerts if datetime.fromisoformat(a["date_created"].replace(' ', 'T')) >= start]
            
        if end_date:
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            filtered_alerts = [a for a in filtered_alerts if datetime.fromisoformat(a["date_created"].replace(' ', 'T')) <= end]
        
        # Calculate metrics
        total_alerts = len(filtered_alerts)
        
        # Group alerts by date for avg calculation
        date_groups = {}
        for alert in filtered_alerts:
            date = alert["date_created"].split()[0]
            if date not in date_groups:
                date_groups[date] = 0
            date_groups[date] += 1
        
        avg_alerts_per_day = sum(date_groups.values()) / len(date_groups) if date_groups else 0
        
        # Calculate distribution for all required variables
        distribution_data = {
            "sensor_type": {},
            "resolution_reason": {},
            "device_type": {},
            "industry": {},
            "event_type": {}
        }
        
        for alert in filtered_alerts:
            for key in distribution_data.keys():
                value = alert[key]
                if value not in distribution_data[key]:
                    distribution_data[key][value] = 0
                distribution_data[key][value] += 1
            
        return {
            "alerts": filtered_alerts,
            "meta": meta,
            "metrics": {
                "total_alerts": total_alerts,
                "avg_alerts_per_day": avg_alerts_per_day,
                "distribution_data": distribution_data
            }
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Alerts data file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")