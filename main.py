# main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os
import time
import logging
from typing import Optional
from functools import lru_cache

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

# Create optimized engine and sessionmaker
# Set echo=False to disable SQL logging and improve performance
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Simple in-memory cache system for improved performance
cache = {}
CACHE_TTL = 300  # Cache time-to-live in seconds (5 minutes)

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

def get_cache_key(filters):
    """Generate a cache key based on filter parameters"""
    return str(hash(str(filters.dict())))

def apply_filters(query, filters):
    """
    Apply all filter conditions to the query.
    This function is extracted to improve code maintainability.
    """
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
    
    # Apply date filters
    if filters.start_date:
        query = query.filter(Alert.date_created >= filters.start_date)
    if filters.end_date:
        query = query.filter(Alert.date_created <= filters.end_date)
    
    return query

def calculate_date_stats(alerts):
    """
    Calculate date-related statistics from alert records.
    More efficient implementation for processing alerts.
    """
    date_groups = {}
    for alert in alerts:
        day = alert.date_created.strftime("%Y-%m-%d")
        date_groups[day] = date_groups.get(day, 0) + 1
    
    avg_per_day = sum(date_groups.values()) / len(date_groups) if date_groups else 0
    
    return {
        "avg_per_day": avg_per_day,
        "date_groups": date_groups
    }

def calculate_distribution_data(alerts):
    """
    Calculate distribution data across different dimensions.
    Optimized to process all fields in a single loop.
    """
    distribution_data = {
        "sensor_type": {},
        "resolution_reason": {},
        "device_type": {},
        "industry": {},
        "event_type": {}
    }
    
    # Process all fields in a single loop for better performance
    for alert in alerts:
        for field in distribution_data:
            value = getattr(alert, field)
            if value:  # Ensure value exists
                distribution_data[field][value] = distribution_data[field].get(value, 0) + 1
    
    return distribution_data

def generate_date_range(start_date, end_date):
    """
    Generate a list of dates between start_date and end_date (inclusive)
    """
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return dates

@app.post("/dashboard_data")
def get_dashboard_data(filters: GetDashboard):
    """
    Endpoint to retrieve dashboard data using provided filters.
    Returns ALL matching records without any limit.
    """
    try:
        # Record start time for performance monitoring
        start_time = time.time()
        
        # Check cache for existing results
        cache_key = get_cache_key(filters)
        if cache_key in cache:
            cached_data, timestamp = cache[cache_key]
            if time.time() - timestamp < CACHE_TTL:
                print(f"Returning cached data (key: {cache_key[:10]}...)")
                return cached_data
        
        with SessionLocal() as session:
            # 1. Get total count of matching records (using optimized COUNT query)
            count_query = session.query(func.count(Alert.id))
            count_query = apply_filters(count_query, filters)
            total_alerts = count_query.scalar() or 0
            
            # 2. Generate the time series data
            time_series_data = []
            
            # If date range is specified, we'll generate time series for the full range
            if filters.start_date and filters.end_date:
                # Create a query that groups alerts by date
                date_query = session.query(
                    func.date(Alert.date_created).label('date'),
                    func.count(Alert.id).label('count')
                ).group_by(func.date(Alert.date_created))
                
                # Apply the same filters to this query
                date_query = apply_filters(date_query, filters)
                
                # Execute the query to get per-date counts
                date_results = date_query.all()
                
                # Convert to dictionary for easy lookup
                date_counts = {result.date.strftime("%Y-%m-%d"): result.count for result in date_results}
                
                # Generate a complete date range (including days with zero counts)
                all_dates = generate_date_range(filters.start_date, filters.end_date)
                
                # Create the time series with counts for every date (0 if no data)
                time_series_data = [
                    {"date": date, "value": date_counts.get(date, 0)}
                    for date in all_dates
                ]
            
            # 3. Query ALL alerts that match the filters - NO LIMIT APPLIED
            print(f"Executing query for ALL alerts. Estimated count: {total_alerts}")
            query = session.query(Alert)
            query = apply_filters(query, filters)
            
            # Add ordering by date
            query = query.order_by(Alert.date_created.desc())
            
            # Execute query to get ALL matching alerts
            alerts = query.all()
            
            alert_count = len(alerts)
            print(f"Retrieved {alert_count} alerts")
            
            # Test some alert dates to ensure they're in the expected format
            if alerts and len(alerts) > 0:
                print(f"Sample alert date format: {alerts[0].date_created}")
                if filters.start_date and filters.end_date:
                    print(f"Date filter range: {filters.start_date} to {filters.end_date}")
            
            # 4. Calculate average alerts per day
            if time_series_data:
                # If we have time series data, use that for the average calculation
                days_with_data = sum(1 for item in time_series_data if item["value"] > 0)
                total_in_series = sum(item["value"] for item in time_series_data)
                
                # Ensure we don't divide by zero
                if days_with_data > 0:
                    avg_alerts_per_day = total_in_series / days_with_data
                else:
                    avg_alerts_per_day = 0
            else:
                # Fall back to calculating from the alerts if no time series
                date_stats = calculate_date_stats(alerts)
                avg_alerts_per_day = date_stats["avg_per_day"]
            
            # 5. Calculate distribution data from ALL alerts
            distribution_data = calculate_distribution_data(alerts)
            
            # Debug distribution data
            for field, values in distribution_data.items():
                print(f"{field} distribution: {len(values)} unique values")
                if values and len(values) > 0:
                    # Print top 3 items
                    top_items = sorted(values.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"Top {field} values: {top_items}")
            
            # 6. Convert alerts to response format
            alerts_response = [alert_to_schema(alert) for alert in alerts]
            
            # 7. Build the response with ALL data
            result = {
                "alerts": alerts_response,
                "metrics": {
                    "total_alerts": total_alerts,
                    "avg_alerts_per_day": avg_alerts_per_day,
                    "distribution_data": distribution_data
                },
                "time_series": time_series_data
            }
            
            # 8. Cache the result
            cache[cache_key] = (result, time.time())
            
            # 9. Log performance statistics
            print(f"Dashboard query processed in {time.time() - start_time:.2f} seconds. Retrieved {alert_count} records, with {len(time_series_data)} time points.")
            
            return result
            
    except Exception as e:
        print(f"Error processing dashboard query: {str(e)}")
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

            # Get ALL alerts - no limit
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

# Additional API endpoint for cache management
@app.post("/clear_cache")
def clear_cache():
    """Clear the API cache (useful for development/testing)"""
    global cache
    old_size = len(cache)
    cache = {}
    return {"message": f"Cleared {old_size} cache entries"}

# Function to clean up expired cache entries
@app.on_event("startup")
async def setup_cache_cleanup():
    """Set up periodic cleaning of expired cache entries"""
    def cleanup_expired_cache():
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [k for k, (_, timestamp) in cache.items() 
                       if current_time - timestamp > CACHE_TTL]
        
        for key in expired_keys:
            cache.pop(key, None)
            
        if expired_keys:
            print(f"Cleaned up {len(expired_keys)} expired cache entries")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")