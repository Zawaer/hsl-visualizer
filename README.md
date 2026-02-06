# HSL Vehicle Visualizer

A tool to create animated videos of Helsinki (HSL) public transport vehicles with fading trails overlaid on a map background.

**Inspired by:** [Pathfinding Simulation](https://www.youtube.com/watch?v=CgW0HPHqFE8) — the glowing paths on dark maps create a cool visual effect.

<img width="1266" height="897" alt="Screenshot 2026-02-03 at 21 43 20" src="https://github.com/user-attachments/assets/f0801f2e-3e4d-41db-b708-8b219ffde7a1" />

## What it does

1. **Fetcher** (`fetcher.py`): Polls the HSL GTFS-RT API every 5 seconds to continuously log vehicle positions (buses, trams, metro, trains, ferries) to CSV.
2. **GTFS Routes** (`fetch_routes.py`): Downloads HSL GTFS package and extracts `routes.txt` for vehicle type detection.
3. **Renderer** (`renderer.py`): Reads CSV data and generates an animation showing each vehicle's movement with fading trails, color-coded by vehicle type. Renders to PNG frames, then encodes to MP4 with distance counter.

## How it works

- **Vehicle type detection** via GTFS `routes.txt` (buses, trams, metro, trains, ferries).
- **Color-coded by vehicle type** (or unique per vehicle—configurable).
- Older trail positions fade over time (configurable duration).
- Latest position highlighted with a dot.
- **Total distance counter** in lower-right corner shows cumulative km traveled.
- Optional background map tiles (CartoDB, OpenStreetMap) with auto-zoom.
- Parallel rendering for speed (~10 fps on M3 MacBook Air).

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   brew install ffmpeg  # For video encoding
   ```

2. **Fetch GTFS routes.txt** (for vehicle type detection):
   ```bash
   python3 fetch_routes.py
   ```
   This downloads the HSL GTFS package and extracts `routes.txt` to your project directory.

3. **Collect vehicle position data:**
   ```bash
   python3 fetcher.py
   ```
   Outputs daily CSVs to `data/` (one per day). **Keep only one CSV** in `data/` when rendering.

## Usage

Edit the `CONFIG` dict at the top of `renderer.py` to set:
- Input/output paths
- Resolution, duration, FPS, trail fade time
- Region (Helsinki+Espoo preset or custom bbox)
- Basemap provider & opacity
- Video encoding settings

Then run:
```bash
python3 renderer.py
```

Done—MP4 output in `output.mp4` (or whatever you set in `CONFIG`).

## Configuration highlights

```python
CONFIG = {
    "input_path": "data",              # CSV directory or single file (MUST be 1 CSV only)
    "output_dir": "frames",            # Frame output directory
    "fps": 25,
    "duration_sec": 30,
    "width_px": 1080,
    "height_px": 1080,
    "use_region": True,                # Crop to preset region?
    "region_name": "helsinki_espoo",   # or define custom in 'regions'
    "use_basemap": True,               # Background map tiles
    "color_by_vehicle_type": False,    # Color by type (bus/tram/metro) vs unique per vehicle
    "encode_video": True,              # Auto-encode to MP4?
    ...
}
```

## Important

- **Only one CSV file** allowed in `data/` folder during rendering (renderer will fail if multiple CSVs are present).
- Delete old CSVs or move them to a backup folder before rendering new videos.
- Don't commit `data/` or `frames/` to git (already in `.gitignore`)—these are generated files and can be hundreds of MB.

## Fetcher

Logs vehicle positions continuously:
```bash
python3 fetcher.py
```

Outputs daily CSVs to `data/` (one per day by default).

## Notes

- Vehicle types detected via GTFS `routes.txt` (run `fetch_routes.py` first).
- Basemap tiles require `contextily` and `pyproj` (included in `requirements.txt`).
- FFmpeg required for video encoding (`brew install ffmpeg` on macOS).
