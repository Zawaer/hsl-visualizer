# HSL Vehicle Visualizer

A tool to create animated videos of Helsinki (HSL) public transport vehicles with fading trails overlaid on a map background.

## What it does

1. **Fetcher** (`fetcher.py`): Polls the HSL GTFS-RT API every 5 seconds to continuously log vehicle positions (buses, trams, metro) to CSV.
2. **Renderer** (`renderer.py`): Reads CSV data and generates an animation showing each vehicle's movement with fading trails. Renders to PNG frames, then encodes to MP4.

## How it works

- Each vehicle gets a unique color (deterministic, same color across frames).
- Older trail positions fade over time (configurable duration).
- Latest position highlighted with a dot.
- Optional background map tiles (CartoDB, OpenStreetMap) with auto-zoom.
- Parallel rendering for speed (~10 fps on M3 MacBook Air).

## Setup

```bash
pip install -r requirements.txt
```

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
    "input_path": "data",              # CSV directory or single file
    "output_dir": "frames",            # Frame output directory
    "fps": 25,
    "duration_sec": 30,
    "width_px": 1080,
    "height_px": 1080,
    "use_region": True,                # Crop to preset region?
    "region_name": "helsinki_espoo",   # or define custom in 'regions'
    "use_basemap": True,               # Background map tiles
    "encode_video": True,              # Auto-encode to MP4?
    ...
}
```

## Fetcher

Logs vehicle positions continuously:
```bash
python3 fetcher.py
```

Outputs daily CSVs to `data/` (one per day by default).

## Notes

- No line IDs or vehicle type detection in first version.
- Basemap tiles require `contextily` and `pyproj` (included in `requirements.txt`).
- FFmpeg required for video encoding.