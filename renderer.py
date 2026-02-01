#!/usr/bin/env python3
# renderer.py
# Reads CSV(s) with columns from fetcher.py and renders fading-trail frames.
# Produces images in frames/ and prints ffmpeg command.
# Now ensures all frames have even width & height for FFmpeg.

import os
import math
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm
from PIL import Image

plt.rcParams['figure.dpi'] = 100

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="CSV file or directory with CSVs (use fetcher output).")
    p.add_argument("--outdir", default="frames",
                   help="Directory to write frame PNGs.")
    p.add_argument("--fps", type=int, default=25, help="Output frames per second.")
    p.add_argument("--duration", type=int, default=30, help="Target video seconds (total).")
    p.add_argument("--width", type=int, default=1080, help="Frame width px.")
    p.add_argument("--height", type=int, default=1080, help="Frame height px.")
    p.add_argument("--trail_seconds", type=int, default=3600,
                   help="How long (seconds) trails remain visible (age fade). Use high value to keep most of day).")
    p.add_argument("--bg_color", default="#0a0a0f", help="Background color.")
    p.add_argument("--point_size", type=float, default=6.0)
    return p.parse_args()

def load_data(input_path):
    if os.path.isdir(input_path):
        files = sorted([os.path.join(input_path,f) for f in os.listdir(input_path) if f.endswith(".csv")])
    else:
        files = [input_path]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["timestamp_fetch_utc"], infer_datetime_format=True)
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
    df["timestamp_fetch_utc"] = pd.to_datetime(df["timestamp_fetch_utc"], utc=True)
    df = df.sort_values("timestamp_fetch_utc")
    return df

def compute_canvas_limits(df, padding=0.01):
    min_lon, max_lon = df["longitude"].min(), df["longitude"].max()
    min_lat, max_lat = df["latitude"].min(), df["latitude"].max()
    lon_pad = (max_lon - min_lon) * padding
    lat_pad = (max_lat - min_lat) * padding
    return (min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad)

def vehicle_color_map(vehicle_ids):
    cmap = plt.get_cmap("tab20")
    mapping = {}
    n = len(vehicle_ids)
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
    """Pad the image to ensure both width & height are even, overwriting the original file."""
    im = Image.open(img_path)
    w, h = im.size
    new_w = w + (w % 2)
    new_h = h + (h % 2)
    if (new_w, new_h) != (w, h):
        new_im = Image.new("RGBA", (new_w, new_h), (0,0,0,0))
        new_im.paste(im, (0,0))
        new_im.save(img_path)
    im.close()

def render_frames(df, outdir, fps=25, duration=30, width=1080, height=1080, trail_seconds=3600, bg_color="#0a0a0f", point_size=6):
    os.makedirs(outdir, exist_ok=True)
    t_start = df["timestamp_fetch_utc"].min()
    t_end = df["timestamp_fetch_utc"].max()
    if t_start == t_end:
        raise RuntimeError("Input CSV has single timestamp only.")
    total_frames = fps * duration
    print("Data time range:", t_start, "→", t_end)
    print("Rendering", total_frames, "frames (fps:", fps, "duration:", duration, "s)")

    frame_times = pd.to_datetime(np.linspace(t_start.value, t_end.value, total_frames)).tz_localize('UTC')
    vehicles = df["vehicle_id"].unique()
    trails = {vid: df[df["vehicle_id"] == vid][["timestamp_fetch_utc","longitude","latitude"]].to_numpy() for vid in vehicles}
    colors = vehicle_color_map(vehicles)
    xmin, xmax, ymin, ymax = compute_canvas_limits(df, padding=0.03)

    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_axes([0,0,1,1])
    ax.set_facecolor(bg_color)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    vehicle_list = list(trails.keys())

    for i, ft in enumerate(tqdm(frame_times, desc="frames")):
        ax.clear()
        ax.set_facecolor(bg_color)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

        for vid in vehicle_list:
            arr = trails[vid]
            times = pd.to_datetime(arr[:,0])
            mask = times <= ft
            if not mask.any():
                continue
            xs = arr[mask,1].astype(float)
            ys = arr[mask,2].astype(float)
            ages = (ft - pd.to_datetime(arr[mask,0])).total_seconds().astype(float)
            alphas = np.clip(1 - (ages / trail_seconds), 0.0, 1.0)
            if len(xs) >= 2:
                segs = trail_segments(xs, ys)
                seg_ages = (ages[:-1] + ages[1:]) / 2.0
                seg_alphas = np.clip(1 - (seg_ages / trail_seconds), 0.0, 1.0)
                lc = LineCollection(segs, linewidths=1.5, colors=[colors[vid]]*len(segs), alpha=1.0, zorder=1)
                seg_rgba = []
                base_color = colors[vid]
                for a in seg_alphas:
                    seg_rgba.append((base_color[0], base_color[1], base_color[2], float(a*0.9)))
                lc.set_colors(seg_rgba)
                ax.add_collection(lc)
            ax.scatter(xs[-1], ys[-1], s=point_size, color=colors[vid], edgecolors='none', zorder=2)

        ts_text = ft.strftime("%Y-%m-%d %H:%M:%S UTC")
        ax.text(0.01, 0.02, ts_text, color="white", fontsize=10, transform=ax.transAxes, zorder=10)

        fname = os.path.join(outdir, f"frame_{i:05d}.png")
        fig.savefig(fname, dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
        save_frame_even_size(fname)  # <-- ensure even dimensions for FFmpeg

    plt.close(fig)
    print("Frames saved to:", outdir)
    print("Use ffmpeg to combine frames into mp4, e.g.:")
    print(f"ffmpeg -y -framerate {fps} -i {outdir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4")

def main():
    args = parse_args()
    df = load_data(args.input)
    render_frames(df, args.outdir, fps=args.fps, duration=args.duration,
                  width=args.width, height=args.height,
                  trail_seconds=args.trail_seconds, bg_color=args.bg_color,
                  point_size=args.point_size)

if __name__ == "__main__":
    main()
