# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os

from models import Base, Alert
from schemas import AlertOut, GetReport, GetDashboard
from mappings import (
    DEVICE_TYPE_MAPPING, EVENT_TYPE_MAPPING, RESOLUTION_REASON_MAPPING,
    INDUSTRY_MAPPING, SENSOR_TYPE_MAPPING, COUNTRY_MAPPING, CONTINENT_MAPPING
)

# Load environment variables
load_dotenv(dotenv_path=".env")
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

# Create synchronous engine and sessionmaker
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=True, bind=engine)

@app.on_event("startup")
def startup_event():
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS map"))
        conn.commit()
    Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Alert Insights Service is up and running!"}

def convert_filter(values, mapping: dict):
    """Convert a list of filter strings into their corresponding int IDs."""
    if values:
        converted = [mapping[val] for val in values if val in mapping]
        return converted if converted else None
    return None

def combine_country_filters(country_list, continent_list):
    """
    Combine country and continent filters.
    Convert the list of country names (if any) and the countries mapped from continents,
    then return a single list of country IDs.
    """
    combined_ids = set()
    if country_list:
        ids = convert_filter(country_list, COUNTRY_MAPPING)
        if ids:
            combined_ids.update(ids)
    if continent_list:
        continent_countries = []
        for cont in continent_list:
            continent_countries.extend(CONTINENT_MAPPING.get(cont, []))
        ids = convert_filter(continent_countries, COUNTRY_MAPPING)
        if ids:
            combined_ids.update(ids)
    return list(combined_ids) if combined_ids else None

def alert_to_schema(alert: Alert) -> AlertOut:
    """Convert an Alert ORM instance into an output model containing only string values."""
    return AlertOut(
        id=alert.id,
        device_type=alert.device_type,
        date_created=alert.date_created,
        sensor_type=alert.sensor_type,
        event_type=alert.event_type,
        resolution_reason=alert.resolution_reason,
        industry=alert.industry,
        longitude=alert.longitude,
        latitude=alert.latitude,
        country=alert.country
    )

@app.post("/dashboard_data")
def get_dashboard_data(filters: GetDashboard):
    """
    Retrieve dashboard data using provided filters. 
    Incoming string filters are converted to int IDs for faster queries, 
    and the response returns only the human-readable strings.
    """
    try:
        with SessionLocal() as session:
            query = session.query(Alert)
            # Apply individual filters
            sensor_ids = convert_filter(filters.sensor_type, SENSOR_TYPE_MAPPING)
            if sensor_ids:
                query = query.filter(Alert.sensor_type_id.in_(sensor_ids))
            industry_ids = convert_filter(filters.industry, INDUSTRY_MAPPING)
            if industry_ids:
                query = query.filter(Alert.industry_id.in_(industry_ids))
            event_ids = convert_filter(filters.event_type, EVENT_TYPE_MAPPING)
            if event_ids:
                query = query.filter(Alert.event_type_id.in_(event_ids))
            resolution_ids = convert_filter(filters.resolution_reason, RESOLUTION_REASON_MAPPING)
            if resolution_ids:
                query = query.filter(Alert.resolution_reason_id.in_(resolution_ids))
            device_ids = convert_filter(filters.device_type, DEVICE_TYPE_MAPPING)
            if device_ids:
                query = query.filter(Alert.device_type_id.in_(device_ids))
            
            # Combine country and continent filters into a single query
            combined_country_ids = combine_country_filters(filters.country, filters.continent)
            if combined_country_ids:
                query = query.filter(Alert.country_id.in_(combined_country_ids))
            
            if filters.start_date:
                query = query.filter(Alert.date_created >= filters.start_date)
            if filters.end_date:
                query = query.filter(Alert.date_created <= filters.end_date)
            
            alerts = query.all()
            total_alerts = len(alerts)
            
            # Compute average alerts per day
            date_groups = {}
            for alert in alerts:
                day = alert.date_created.strftime("%Y-%m-%d")
                date_groups[day] = date_groups.get(day, 0) + 1
            avg_alerts_per_day = sum(date_groups.values()) / len(date_groups) if date_groups else 0

            # Compute distribution counts (using only string fields)
            distribution_data = {
                "sensor_type": {},
                "resolution_reason": {},
                "device_type": {},
                "industry": {},
                "event_type": {}
            }
            for alert in alerts:
                distribution_data["sensor_type"][alert.sensor_type] = distribution_data["sensor_type"].get(alert.sensor_type, 0) + 1
                distribution_data["resolution_reason"][alert.resolution_reason] = distribution_data["resolution_reason"].get(alert.resolution_reason, 0) + 1
                distribution_data["device_type"][alert.device_type] = distribution_data["device_type"].get(alert.device_type, 0) + 1
                distribution_data["industry"][alert.industry] = distribution_data["industry"].get(alert.industry, 0) + 1
                distribution_data["event_type"][alert.event_type] = distribution_data["event_type"].get(alert.event_type, 0) + 1
            
            alerts_response = [alert_to_schema(alert) for alert in alerts]
            
            return {
                "alerts": alerts_response,
                "metrics": {
                    "total_alerts": total_alerts,
                    "avg_alerts_per_day": avg_alerts_per_day,
                    "distribution_data": distribution_data
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/report_data")
def get_report_data(filters: GetReport):
    """
    Retrieve report data within a date range and optional filters.
    Depending on the date range:
      - Under 3 weeks: group by day
      - Between 3 and 12 weeks: group by week (returns the Monday of each week)
      - Over 12 weeks: group by month (returns the first day of the month)
    For each grouping field (resolution_reason, device_type, sensor_type, industry, event_type),
    any category whose total count is under 10% of the maximum is combined into an "other" group.
    Even if one day/week/month is empty, the response will include it with a 0 (or an empty JSON for categories).
    """
    try:
        with SessionLocal() as session:
            # Build the base query with date filters and any additional filters.
            query = session.query(Alert)
            query = query.filter(
                Alert.date_created >= filters.date_start,
                Alert.date_created <= filters.date_end
            )
            res_ids = convert_filter(filters.resolution_reason, RESOLUTION_REASON_MAPPING)
            if res_ids:
                query = query.filter(Alert.resolution_reason_id.in_(res_ids))
            dev_ids = convert_filter(filters.device_type, DEVICE_TYPE_MAPPING)
            if dev_ids:
                query = query.filter(Alert.device_type_id.in_(dev_ids))
            sens_ids = convert_filter(filters.sensor_type, SENSOR_TYPE_MAPPING)
            if sens_ids:
                query = query.filter(Alert.sensor_type_id.in_(sens_ids))
            event_ids = convert_filter(filters.event_type, EVENT_TYPE_MAPPING)
            if event_ids:
                query = query.filter(Alert.event_type_id.in_(event_ids))
            ind_ids = convert_filter(filters.industry, INDUSTRY_MAPPING)
            if ind_ids:
                query = query.filter(Alert.industry_id.in_(ind_ids))

            # Combine country and continent filters if provided.
            combined_country_ids = combine_country_filters(filters.country, filters.continent)
            if combined_country_ids:
                query = query.filter(Alert.country_id.in_(combined_country_ids))

            alerts = query.all()

            # Determine grouping frequency based on the date range in days.
            delta_days = (filters.date_end - filters.date_start).days
            if delta_days < 21:
                freq = "D"  # daily
            elif delta_days < 84:
                freq = "W"  # weekly
            else:
                freq = "M"  # monthly

            # Helper function: get bucket label for an alert based on the frequency.
            def get_bucket(dt):
                if freq == "D":
                    return dt.strftime("%Y-%m-%d")
                elif freq == "W":
                    # Return the Monday of the week.
                    monday = dt - timedelta(days=dt.weekday())
                    return monday.strftime("%Y-%m-%d")
                elif freq == "M":
                    # Return the first day of the month.
                    month_start = dt.replace(day=1)
                    return month_start.strftime("%Y-%m-%d")

            # Generate all expected time buckets between date_start and date_end.
            expected_buckets = set()
            if freq == "D":
                step = timedelta(days=1)
                current = filters.date_start
                while current <= filters.date_end:
                    expected_buckets.add(current.strftime("%Y-%m-%d"))
                    current += step
            elif freq == "W":
                # For weekly, start with the Monday of the week containing date_start.
                current = filters.date_start - timedelta(days=filters.date_start.weekday())
                step = timedelta(weeks=1)
                # Continue until we pass the end date.
                while current <= filters.date_end:
                    expected_buckets.add(current.strftime("%Y-%m-%d"))
                    current += step
            elif freq == "M":
                # For monthly, start with the first day of the month of date_start.
                current = filters.date_start.replace(day=1)
                while current <= filters.date_end:
                    expected_buckets.add(current.strftime("%Y-%m-%d"))
                    # Advance one month.
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1)
                    else:
                        current = current.replace(month=current.month + 1)
            
            # Build overall time series data grouped by the chosen frequency.
            overall_series = {bucket: 0 for bucket in expected_buckets}
            for alert in alerts:
                bucket = get_bucket(alert.date_created)
                overall_series[bucket] = overall_series.get(bucket, 0) + 1

            sorted_buckets = sorted(overall_series.keys())
            time_series_overall = {
                "date_created": sorted_buckets,
                "alert_count": [overall_series[bucket] for bucket in sorted_buckets]
            }

            # Prepare grouping for each categorical field with all expected buckets.
            grouping_fields = ["resolution_reason", "device_type", "sensor_type", "industry", "event_type"]
            grouped_data = {field: {bucket: {} for bucket in expected_buckets} for field in grouping_fields}

            # For each alert, group the counts by the chosen time bucket and category.
            for alert in alerts:
                bucket = get_bucket(alert.date_created)
                for field in grouping_fields:
                    # Assumes that the Alert object has an attribute matching the field name.
                    category = getattr(alert, field)
                    if bucket not in grouped_data[field]:
                        grouped_data[field][bucket] = {}
                    grouped_data[field][bucket][category] = grouped_data[field][bucket].get(category, 0) + 1

            # For each grouping field, combine any categories with overall count less than 10% of the max.
            for field in grouping_fields:
                # Compute total counts per category across all time buckets.
                category_totals = {}
                for bucket, counts in grouped_data[field].items():
                    for category, count in counts.items():
                        category_totals[category] = category_totals.get(category, 0) + count

                if category_totals:
                    max_total = max(category_totals.values())
                    threshold = 0.1 * max_total
                    # Identify categories to combine.
                    categories_to_combine = [cat for cat, total in category_totals.items() if total < threshold]

                    # For each time bucket, sum up counts for categories under the threshold.
                    for bucket, counts in grouped_data[field].items():
                        other_count = 0
                        new_counts = {}
                        for category, count in counts.items():
                            if category in categories_to_combine:
                                other_count += count
                            else:
                                new_counts[category] = count
                        if other_count > 0:
                            new_counts["other"] = new_counts.get("other", 0) + other_count
                        grouped_data[field][bucket] = new_counts

            return {
                "time_series_overall": time_series_overall,
                "grouped_data": grouped_data,
                "grouping_frequency": freq  # Optionally include the frequency used.
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")