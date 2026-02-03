import time
import csv
import os
import signal
from datetime import datetime, timezone
import requests
from google.transit import gtfs_realtime_pb2

from config import FETCHER as CONFIG

running = True

def signal_handler(signum, frame):
    global running
    running = False
    print("Stopping fetcher...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def current_csv_path():
    date = datetime.now().strftime("%Y%m%d")
    fname = f"vehicle_positions_{date}.csv"
    return os.path.join(CONFIG["output_dir"], fname)

# CSV columns to write (fixed)
CSV_COLUMNS = [
    "timestamp_fetch_utc",
    "vehicle_id",
    "route_id",
    "latitude",
    "longitude",
    "bearing",
    "speed",
    "current_status",
    "occupancy_status",
    "trip_start_time",
    "raw_timestamp",
]

def ensure_csv_header(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

def extract_field_safe(obj, field, default=""):
    try:
        return getattr(obj, field)
    except Exception:
        return default

def poll_once(csv_path):
    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        resp = requests.get(CONFIG["vehicle_pos_url"], timeout=10)
        resp.raise_for_status()
        feed.ParseFromString(resp.content)
    except Exception as e:
        print("Fetch error:", e)
        return 0

    rows = []
    now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle

        vehicle_id = ""
        try:
            # v.vehicle is a nested message with id field sometimes
            vehicle_id = v.vehicle.id if v.vehicle and v.vehicle.id else ""
        except Exception:
            vehicle_id = ""

        route_id = ""
        try:
            route_id = v.trip.route_id if v.trip and v.trip.route_id else ""
        except Exception:
            route_id = ""

        lat = getattr(v.position, "latitude", "")
        lon = getattr(v.position, "longitude", "")
        bearing = getattr(v.position, "bearing", "")
        # speed may be present in different implementations; try position.speed or vehicle.speed
        speed = getattr(v.position, "speed", "") or getattr(v, "speed", "")
        current_status = getattr(v, "current_status", "")
        occupancy_status = getattr(v, "occupancy_status", "")
        trip_start_time = getattr(v.trip, "start_time", "") if v.trip is not None else ""
        raw_ts = getattr(v, "timestamp", "")

        rows.append([
            now,
            str(vehicle_id),
            str(route_id),
            lat,
            lon,
            bearing,
            speed,
            current_status,
            occupancy_status,
            trip_start_time,
            raw_ts
        ])

    if rows:
        # append rows to CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    return len(rows)

def main_loop():
    print("Starting fetcher. Poll interval:", CONFIG["poll_interval"], "s")
    current_path = current_csv_path()
    ensure_csv_header(current_path)

    while running:
        path = current_csv_path()
        if path != current_path:
            # date rolled over: start a new file
            current_path = path
            ensure_csv_header(current_path)
            print("Rotated to new CSV:", current_path)

        n = poll_once(current_path)
        if n:
            print(f"Logged {n} positions at {datetime.utcnow().isoformat()} to {current_path}")
        else:
            print(f"No positions logged at {datetime.utcnow().isoformat()}")
        # sleep with small increments so SIGINT is responsive
        for _ in range(int(CONFIG["poll_interval"]*10)):
            if not running:
                break
            time.sleep(0.1)

    print("Fetcher stopped.")

if __name__ == "__main__":
    main_loop()
