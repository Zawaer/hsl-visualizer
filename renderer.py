import os
import math
import signal
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageEnhance
from multiprocessing import Pool, cpu_count
from io import BytesIO

try:
    from pyproj import Transformer
except Exception:  # optional dependency
    Transformer = None

try:
    import contextily as ctx
except Exception:  # optional dependency
    ctx = None

from config import RENDERER as CONFIG

plt.rcParams['figure.dpi'] = 100

# ============================================================================

_WEBMERCATOR_INITIAL_RES_M_PER_PX = 156543.03392804097  # at equator, zoom=0
_WEB_TILE_SIZE_PX = 256
_BASEMAP_MAX_TILES_DEFAULT = 256

def encode_video_ffmpeg(frames_dir: str, fps: int, output_path: str, crf: int = 18, preset: str = "medium", overwrite: bool = False):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`) or run FFmpeg manually.")

    input_pattern = os.path.join(frames_dir, "frame_%05d.png")
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-movflags",
        "+faststart",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_path,
    ]

    print("Encoding video with FFmpeg:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def load_data(input_path):
    if os.path.isdir(input_path):
        files = sorted([os.path.join(input_path,f) for f in os.listdir(input_path) if f.endswith(".csv")])
    else:
        files = [input_path]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["timestamp_fetch_utc"])
            dfs.append(df)
        except Exception as e:
            print("Failed to read", f, ":", e)
    if not dfs:
        raise RuntimeError("No CSVs loaded.")
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.dropna(subset=["latitude","longitude"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude","longitude"])

    # Basic sanity filtering to avoid wild outliers breaking bounds/basemaps
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]
    df = df[~((df["latitude"] == 0) & (df["longitude"] == 0))]

    df["timestamp_fetch_utc"] = pd.to_datetime(df["timestamp_fetch_utc"], utc=True)
    df = df.sort_values("timestamp_fetch_utc")
    return df

def compute_canvas_limits(df, padding=0.01):
    min_lon, max_lon = df["longitude"].min(), df["longitude"].max()
    min_lat, max_lat = df["latitude"].min(), df["latitude"].max()
    lon_pad = (max_lon - min_lon) * padding
    lat_pad = (max_lat - min_lat) * padding
    return (min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad)


def region_bbox(name: str):
    """Return (min_lon, max_lon, min_lat, max_lat) for known presets."""
    presets = CONFIG["regions"]
    return presets.get(name)


def filter_df_to_bbox(df: pd.DataFrame, bbox):
    min_lon, max_lon, min_lat, max_lat = bbox
    return df[
        (df["longitude"].between(min_lon, max_lon))
        & (df["latitude"].between(min_lat, max_lat))
    ]


def compute_canvas_limits_xy(df, x_col="x", y_col="y", padding=0.01):
    min_x, max_x = df[x_col].min(), df[x_col].max()
    min_y, max_y = df[y_col].min(), df[y_col].max()
    x_pad = (max_x - min_x) * padding
    y_pad = (max_y - min_y) * padding
    return (min_x - x_pad, max_x + x_pad, min_y - y_pad, max_y + y_pad)


def _resolve_basemap_provider(provider_str: str):
    """Resolve a basemap provider string into a contextily provider object."""
    if ctx is None:
        return None

    key = provider_str.strip()
    shortcuts = {
        "cartodb_positron": "CartoDB.Positron",
        "cartodb_darkmatter": "CartoDB.DarkMatter",
        "osm": "OpenStreetMap.Mapnik",
    }
    key = shortcuts.get(key.lower(), key)

    # If user passed a dotted providers path like "CartoDB.Positron"
    if "." in key:
        cur = ctx.providers
        for part in key.split("."):
            cur = cur[part]
        return cur

    return None


def _ensure_deps_for_basemap():
    if ctx is None:
        raise RuntimeError("Basemap requested but 'contextily' is not installed. Install: pip install contextily")
    if Transformer is None:
        raise RuntimeError("Basemap requested but 'pyproj' is not installed. Install: pip install pyproj")


def _project_df_to_web_mercator(df: pd.DataFrame) -> pd.DataFrame:
    """Adds x/y columns in EPSG:3857 for plotting over web tile basemaps."""
    _ensure_deps_for_basemap()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    out = df.copy()
    out["x"] = xs
    out["y"] = ys
    return out


def _project_bbox_to_web_mercator(bbox):
    """Project lon/lat bbox to EPSG:3857 extents (xmin, xmax, ymin, ymax)."""
    _ensure_deps_for_basemap()
    min_lon, max_lon, min_lat, max_lat = bbox
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    corners_lon = np.array([min_lon, min_lon, max_lon, max_lon], dtype=float)
    corners_lat = np.array([min_lat, max_lat, min_lat, max_lat], dtype=float)
    xs, ys = transformer.transform(corners_lon, corners_lat)
    return float(np.min(xs)), float(np.max(xs)), float(np.min(ys)), float(np.max(ys))


def _prepare_basemap_npz(npz_path: str, xmin: float, xmax: float, ymin: float, ymax: float, provider, zoom: int):
    """Fetch and cache basemap tiles for the given Web Mercator bounds."""
    _ensure_deps_for_basemap()
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)

    img, ext = ctx.bounds2img(
        xmin,
        ymin,
        xmax,
        ymax,
        zoom=zoom,
        source=provider,
        ll=False,
        n_connections=4,
        max_retries=2,
        wait=0,
    )
    extent = np.array([ext[0], ext[1], ext[2], ext[3]], dtype=float)
    np.savez_compressed(npz_path, img=img, extent=extent)


def _parse_zoom(value):
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s in ("auto", "a"):
        return "auto"
    return int(s)


def _auto_basemap_zoom_for_extent(xmin: float, xmax: float, ymin: float, ymax: float, width_px: int, height_px: int, lat_center_deg: float):
    """Choose a tile zoom so the basemap has roughly >= canvas resolution."""
    bbox_w_m = max(1.0, float(xmax - xmin))
    bbox_h_m = max(1.0, float(ymax - ymin))
    target_res_x = bbox_w_m / max(1, int(width_px))
    target_res_y = bbox_h_m / max(1, int(height_px))
    target_res = min(target_res_x, target_res_y)

    lat_center_rad = math.radians(float(lat_center_deg))
    cos_lat = max(0.1, abs(math.cos(lat_center_rad)))

    # resolution(z) = cos(lat) * initial_res / 2^z
    zoom_float = math.log2((cos_lat * _WEBMERCATOR_INITIAL_RES_M_PER_PX) / max(1e-9, target_res))
    zoom = int(max(0, min(19, math.ceil(zoom_float))))

    def tiles_for_zoom(z: int) -> int:
        res = (cos_lat * _WEBMERCATOR_INITIAL_RES_M_PER_PX) / (2 ** z)
        tile_span_m = res * _WEB_TILE_SIZE_PX
        tiles_x = int(math.ceil(bbox_w_m / tile_span_m))
        tiles_y = int(math.ceil(bbox_h_m / tile_span_m))
        return max(1, tiles_x) * max(1, tiles_y)

    # Cap the number of tiles so we don't accidentally download a massive mosaic.
    while zoom > 0 and tiles_for_zoom(zoom) > _BASEMAP_MAX_TILES_DEFAULT:
        zoom -= 1

    return zoom


def _run_with_timeout(seconds: int, fn, *args, **kwargs):
    """Run fn with a hard timeout (Unix only). Returns (ok, result_or_exc)."""
    if seconds <= 0:
        try:
            return True, fn(*args, **kwargs)
        except Exception as e:
            return False, e

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        return True, fn(*args, **kwargs)
    except Exception as e:
        return False, e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


_BASEMAP_CACHE = {}


def _load_basemap_cached(npz_path: str):
    cached = _BASEMAP_CACHE.get(npz_path)
    if cached is not None:
        return cached
    data = np.load(npz_path)
    img = data["img"]
    extent = data["extent"].astype(float)
    _BASEMAP_CACHE[npz_path] = (img, extent)
    return img, extent

def vehicle_color_map(vehicle_ids):
    cmap = plt.get_cmap("tab20")
    mapping = {}
    for i, vid in enumerate(sorted(vehicle_ids)):
        mapping[vid] = cmap(i % 20)
    return mapping

def trail_segments(xs, ys):
    pts = np.array([xs, ys]).T
    if len(pts) < 2:
        return np.empty((0,2,2))
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    return segs

def save_frame_even_size(img_path):
    im = Image.open(img_path)
    w, h = im.size
    new_w = w + (w % 2)
    new_h = h + (h % 2)
    if (new_w, new_h) != (w, h):
        new_im = Image.new("RGBA", (new_w, new_h), (0,0,0,0))
        new_im.paste(im, (0,0))
        new_im.save(img_path)
    im.close()

def _fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image (RGBA)."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert('RGBA')
    return img

def _render_trails_only(vehicle_list, trails, colors, ft, trail_seconds, xmin, xmax, ymin, ymax, width, height):
    """Render trails and dots on transparent background, return PIL Image."""
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_axes([0,0,1,1])
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    for vid in vehicle_list:
        arr = trails[vid]
        times = pd.to_datetime(arr[:,0])
        mask = times <= ft
        if not mask.any():
            continue
        xs = arr[mask,1].astype(float)
        ys = arr[mask,2].astype(float)
        ages = (ft - pd.to_datetime(arr[mask,0])).total_seconds().astype(float)
        if len(xs) >= 2:
            segs = trail_segments(xs, ys)
            seg_ages = (ages[:-1] + ages[1:]) / 2.0
            seg_alphas = np.clip(1 - (seg_ages / trail_seconds), 0.0, 1.0)
            base_color = colors[vid]
            seg_rgba = []
            for a in seg_alphas:
                seg_rgba.append((base_color[0], base_color[1], base_color[2], float(a*0.9)))
            lc = LineCollection(segs, linewidths=CONFIG["trail_width"], alpha=1.0, zorder=1)
            lc.set_colors(seg_rgba)
            ax.add_collection(lc)
        ax.scatter(xs[-1], ys[-1], s=CONFIG.get("point_size", 16), color=colors[vid], edgecolors='none', zorder=2)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, transparent=True, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert('RGBA')
    plt.close(fig)
    return img

def render_single_frame(args):
    """Render a single frame (used by worker pool)."""
    (
        i,
        ft,
        trails,
        colors,
        vehicle_list,
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        bg_color,
        point_size,
        trail_seconds,
        outdir,
        basemap_npz,
        basemap_alpha,
        basemap_interpolation,
    ) = args

    # -- Step 1: Create base layer (background + basemap) --
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_axes([0,0,1,1])
    ax.set_facecolor(bg_color)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')

    if basemap_npz:
        try:
            bm_img, bm_extent = _load_basemap_cached(basemap_npz)
            ax.imshow(bm_img, extent=bm_extent.tolist(), interpolation=basemap_interpolation, zorder=0, alpha=basemap_alpha)
        except Exception:
            pass

    # Save base layer to PIL image
    base_img = _fig_to_pil(fig)
    plt.close(fig)

    # -- Step 2: Render trails on transparent background --
    trails_img = _render_trails_only(vehicle_list, trails, colors, ft, trail_seconds, xmin, xmax, ymin, ymax, width, height)
    
    # Resize trails_img to match base_img if needed
    if trails_img.size != base_img.size:
        trails_img = trails_img.resize(base_img.size, Image.LANCZOS)

    # -- Step 3: Create glow layer via Gaussian blur --
    if CONFIG.get("trail_glow", False):
        glow_blur_radius = CONFIG.get("trail_glow_blur_radius", 15)  # in pixels
        glow_intensity = CONFIG.get("trail_glow_intensity", 1.5)  # brightness boost
        
        # Convert to numpy for premultiplied alpha blurring (preserves colors)
        img_arr = np.array(trails_img, dtype=np.float32)
        r = img_arr[:, :, 0]
        g = img_arr[:, :, 1]
        b = img_arr[:, :, 2]
        a = img_arr[:, :, 3]
        
        # Convert to premultiplied alpha: RGB = RGB * (A / 255)
        alpha_norm = a / 255.0
        r_premult = r * alpha_norm
        g_premult = g * alpha_norm
        b_premult = b * alpha_norm
        
        # Apply Gaussian blur to all channels (in premultiplied space)
        from scipy.ndimage import gaussian_filter
        sigma = glow_blur_radius / 2.0  # Convert radius to sigma
        for _ in range(2):
            r_premult = gaussian_filter(r_premult, sigma=sigma)
            g_premult = gaussian_filter(g_premult, sigma=sigma)
            b_premult = gaussian_filter(b_premult, sigma=sigma)
            a = gaussian_filter(a, sigma=sigma)
        
        # Un-premultiply: RGB = RGB / (A / 255)
        alpha_norm = a / 255.0
        alpha_safe = np.maximum(alpha_norm, 0.001)  # Avoid division by zero
        r_out = np.clip(r_premult / alpha_safe, 0, 255)
        g_out = np.clip(g_premult / alpha_safe, 0, 255)
        b_out = np.clip(b_premult / alpha_safe, 0, 255)
        
        # Boost alpha for glow visibility
        a_out = np.clip(a * glow_intensity, 0, 255)
        
        # Rebuild glow image
        glow_arr = np.stack([r_out, g_out, b_out, a_out], axis=-1).astype(np.uint8)
        glow_img = Image.fromarray(glow_arr, mode='RGBA')
        
        # Composite: base -> glow -> trails
        composite = base_img.copy()
        composite = Image.alpha_composite(composite, glow_img)
        composite = Image.alpha_composite(composite, trails_img)
    else:
        # No glow, just composite trails on base
        composite = Image.alpha_composite(base_img, trails_img)

    # -- Step 4: Add timestamp --
    # Create a small figure just for the text overlay
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_axes([0,0,1,1])
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ts_text = ft.strftime("%Y-%m-%d %H:%M:%S UTC")
    ax.text(0.01, 0.02, ts_text, color="white", fontsize=10, transform=ax.transAxes)
    
    text_img = _fig_to_pil(fig)
    plt.close(fig)
    
    # Resize text overlay if needed
    if text_img.size != composite.size:
        text_img = text_img.resize(composite.size, Image.LANCZOS)
    
    final_img = Image.alpha_composite(composite, text_img)

    # -- Step 5: Save final frame --
    fname = os.path.join(outdir, f"frame_{i:05d}.png")
    os.makedirs(outdir, exist_ok=True)
    final_img.save(fname, 'PNG')
    save_frame_even_size(fname)

def render_frames_parallel(
    df,
    fps=25,
    duration=30,
    width=1080,
    height=1080,
    trail_seconds=3600,
    bg_color="#0a0a0f",
    point_size=6,
    workers=cpu_count(),
    outdir="frames",
    region="auto",
    bbox=None,
    filter_outside_bbox=False,
    basemap=False,
    basemap_provider="cartodb_positron",
    basemap_zoom="auto",
    basemap_alpha=0.85,
    basemap_prefetch_timeout=90,
    basemap_interpolation="bilinear",
):
    # Decide framing bbox (lon/lat)
    effective_bbox = None
    if bbox is not None:
        effective_bbox = tuple(bbox)
    elif region and region != "auto":
        effective_bbox = region_bbox(region)

    if effective_bbox is not None and filter_outside_bbox:
        df = filter_df_to_bbox(df, effective_bbox)
        if df.empty:
            raise RuntimeError("No data left after bbox filtering. Try widening bbox or disabling --filter_outside_bbox")

    t_start = df["timestamp_fetch_utc"].min()
    t_end = df["timestamp_fetch_utc"].max()
    total_frames = fps * duration
    print("Data time range:", t_start, "→", t_end)
    print(f"Rendering {total_frames} frames at {fps} fps using {workers} workers.")

    frame_times = pd.to_datetime(np.linspace(t_start.value, t_end.value, total_frames)).tz_localize('UTC')

    basemap_npz = None
    work_df = df
    x_col, y_col = "longitude", "latitude"
    if basemap:
        _ensure_deps_for_basemap()
        work_df = _project_df_to_web_mercator(df)
        x_col, y_col = "x", "y"

    vehicles = work_df["vehicle_id"].unique()
    trails = {vid: work_df[work_df["vehicle_id"] == vid][["timestamp_fetch_utc", x_col, y_col]].to_numpy() for vid in vehicles}
    colors = vehicle_color_map(vehicles)

    if basemap:
        if effective_bbox is not None:
            xmin, xmax, ymin, ymax = _project_bbox_to_web_mercator(effective_bbox)
        else:
            xmin, xmax, ymin, ymax = compute_canvas_limits_xy(work_df, x_col=x_col, y_col=y_col, padding=0.03)

        if effective_bbox is not None:
            lat_center = (effective_bbox[2] + effective_bbox[3]) / 2.0
        else:
            lat_center = float(df["latitude"].median()) if "latitude" in df.columns else 60.2

        provider_obj = _resolve_basemap_provider(basemap_provider)
        if provider_obj is None:
            raise RuntimeError(f"Unknown basemap provider: {basemap_provider}")

        zoom = _parse_zoom(basemap_zoom)
        if zoom == "auto":
            zoom = _auto_basemap_zoom_for_extent(xmin, xmax, ymin, ymax, width, height, lat_center_deg=lat_center)
            print(f"Basemap: auto-zoom selected {zoom} for {width}x{height}")

        basemap_npz = os.path.join(outdir, f"_basemap_cache_z{zoom}.npz")
        if not os.path.exists(basemap_npz):
            print(f"Basemap: downloading tiles (provider={basemap_provider}, zoom={zoom}) ...")
            t0 = time.time()
            ok, res = _run_with_timeout(
                basemap_prefetch_timeout,
                _prepare_basemap_npz,
                basemap_npz,
                xmin,
                xmax,
                ymin,
                ymax,
                provider_obj,
                zoom,
            )
            if ok:
                print(f"Basemap: cached to {basemap_npz} in {time.time() - t0:.1f}s")
            else:
                print(f"Basemap: failed ({res}). Falling back to solid background.")
                basemap_npz = None
    else:
        if effective_bbox is not None:
            min_lon, max_lon, min_lat, max_lat = effective_bbox
            xmin, xmax, ymin, ymax = min_lon, max_lon, min_lat, max_lat
        else:
            xmin, xmax, ymin, ymax = compute_canvas_limits(df, padding=0.03)

    vehicle_list = list(trails.keys())

    # Prepare args for all frames
    args_list = [(
        i,
        ft,
        trails,
        colors,
        vehicle_list,
        xmin,
        xmax,
        ymin,
        ymax,
        width,
        height,
        bg_color,
        point_size,
        trail_seconds,
        outdir,
        basemap_npz,
        basemap_alpha,
        basemap_interpolation,
    )
                 for i, ft in enumerate(frame_times)]

    with Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(render_single_frame, args_list), total=total_frames, desc="frames"))

    print(f"Frames saved to {outdir}/")
    print("To encode into MP4 manually:")
    print(f"ffmpeg -y -framerate {fps} -i {outdir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4")

def main():
    # Load data
    df = load_data(CONFIG["input_path"])
    
    # Determine effective bbox/region
    effective_bbox = None
    if CONFIG["use_region"]:
        effective_bbox = region_bbox(CONFIG["region_name"])
    
    # Render frames
    render_frames_parallel(
        df,
        fps=CONFIG["fps"],
        duration=CONFIG["duration_sec"],
        width=CONFIG["width_px"],
        height=CONFIG["height_px"],
        trail_seconds=CONFIG["trail_seconds"],
        bg_color=CONFIG["bg_color"],
        point_size=CONFIG["point_size"],
        workers=CONFIG["num_workers"],
        outdir=CONFIG["output_dir"],
        region=CONFIG["region_name"] if CONFIG["use_region"] else "auto",
        bbox=effective_bbox,
        filter_outside_bbox=CONFIG["filter_outside_region"],
        basemap=CONFIG["use_basemap"],
        basemap_provider=CONFIG["basemap_provider"],
        basemap_zoom=CONFIG["basemap_zoom"],
        basemap_alpha=CONFIG["basemap_alpha"],
        basemap_prefetch_timeout=CONFIG["basemap_prefetch_timeout"],
        basemap_interpolation=CONFIG["basemap_interpolation"],
    )
    
    # Encode to MP4 if requested
    if CONFIG["encode_video"]:
        encode_video_ffmpeg(
            CONFIG["output_dir"],
            fps=CONFIG["fps"],
            output_path=CONFIG["video_output_path"],
            crf=CONFIG["video_crf"],
            preset=CONFIG["video_preset"],
            overwrite=CONFIG["video_overwrite"],
        )
        # Delete individual PNG frames after successful encoding (keep basemap cache)
        if CONFIG["cleanup_frames"]:
            for name in os.listdir(CONFIG["output_dir"]):
                if name.startswith("frame_") and name.endswith(".png"):
                    try:
                        os.remove(os.path.join(CONFIG["output_dir"], name))
                    except Exception:
                        pass
            print(f"Cleaned up PNG frames. Video: {CONFIG['video_output_path']}")

if __name__ == "__main__":
    main()
