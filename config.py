import os
from multiprocessing import cpu_count
#TEST
# ============================================================================
# FETCHER CONFIGURATION
# ============================================================================

FETCHER = {
    # GTFS-RT API endpoint
    "vehicle_pos_url": "https://realtime.hsl.fi/realtime/vehicle-positions/v2/hsl",
    
    # Poll interval in seconds
    "poll_interval": 5,
    
    # Output directory for CSV logs
    "output_dir": "data",
}

# ============================================================================
# RENDERER CONFIGURATION
# ============================================================================

RENDERER = {
    # ========== Input/Output ==========
    "input_path": "data",  # CSV file or directory with CSVs
    "output_dir": "frames",  # Directory to write frame PNGs
    
    # ========== Rendering params ==========
    "fps": 25,
    "duration_sec": 3,
    "width_px": 2160,
    "height_px": 2160,
    "trail_seconds": 300,
    "trail_width": 4,  # Trail line width in pixels
    "color_by_vehicle_type": True,  # Color trails by vehicle type (bus, tram, metro) instead of unique per vehicle
    "vehicle_type_colors": {
        "bus": (0.2, 0.6, 1.0),       # Blue
        "tram": (0.2, 0.9, 0.3),      # Green
        "metro": (1.0, 0.2, 0.2),     # Red
        "train": (0.7, 0.2, 0.9),     # Purple
        "ferry": (0.1, 0.8, 0.8),     # Cyan
        "other": (1.0, 1.0, 1.0),     # White
    },
    "trail_glow": True,  # Enable glow effect around trails?
    "trail_glow_blur_radius": 20,  # Gaussian blur radius in pixels (higher = wider glow)
    "trail_glow_intensity": 2.0,  # Glow brightness multiplier (1.0 = same as trail, 2.0 = 2x brighter)
    "bg_color": "#0a0a0f",
    "point_size": 16.0,
    "num_workers": max(1, cpu_count() - 1),  # Leave 1 core free for system responsiveness
    
    # ========== Framing / Region ==========
    "use_region": True,  # Use a preset region or auto-bounds?
    "region_name": "helsinki_espoo",  # "helsinki_espoo" or define custom in 'regions'
    "regions": {
        "helsinki_espoo": (24.6, 25.2, 60.12, 60.34),  # min_lon, max_lon, min_lat, max_lat
    },
    "filter_outside_region": True,  # Drop GPS points outside region?
    
    # ========== Basemap (background map tiles) ==========
    "use_basemap": True,  # Enable background tiles?
    "basemap_provider": "cartodb_darkmatter_nolabels",  # Options: cartodb_positron, cartodb_positron_nolabels, cartodb_darkmatter, cartodb_darkmatter_nolabels, osm
    "basemap_alpha": 0.8,  # Opacity: 0.0 (transparent) to 1.0 (opaque)
    "basemap_interpolation": "bilinear",  # "nearest", "bilinear", "bicubic"
    "basemap_zoom": "auto",  # "auto" to pick based on resolution, or explicit int (e.g., 14)
    "basemap_prefetch_timeout": 90,  # Max seconds waiting for tiles before fallback
    
    # ========== Video encoding ==========
    "encode_video": True,  # Automatically encode MP4 after rendering?
    "video_output_path": "output.mp4",  # Output MP4 file
    "video_crf": 18,  # Quality: lower=better (0-51), typical 18-28
    "video_preset": "medium",  # Speed: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
    "video_overwrite": True,  # Overwrite output MP4 if it exists?
    "cleanup_frames": True,  # Delete PNG frames after successful video encode?
}

# ============================================================================
# Ensure output directories exist
# ============================================================================

os.makedirs(FETCHER["output_dir"], exist_ok=True)
os.makedirs(RENDERER["output_dir"], exist_ok=True)
