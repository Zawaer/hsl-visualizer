HSL vehicle visualizer

Renderer supports an optional map (tile) basemap background.

Install basemap deps:

	pip install -r requirements.txt

Or (minimum for basemap only):

	pip install contextily pyproj

Example render with basemap:

	python3 renderer.py --input data.csv --outdir frames --basemap --basemap_provider cartodb_positron --basemap_zoom 12

Keep framing fixed to Helsinki+Espoo and ignore outlier GPS points:

	python3 renderer.py --input data --outdir frames --basemap --region helsinki_espoo --filter_outside_bbox

Or set an explicit bbox (min_lon max_lon min_lat max_lat):

	python3 renderer.py --input data --outdir frames --basemap --bbox 24.30 25.35 60.05 60.35 --filter_outside_bbox

FFmpeg:

	ffmpeg -y -framerate 25 -i frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4

One-shot render + encode (requires `ffmpeg` installed):

	python3 renderer.py --input data --outdir frames --basemap --region helsinki_espoo --filter_outside_bbox --video --video_out output.mp4 --video_overwrite

Notes on auto-encoding:
- Good for convenience/repeatability.
- Kept optional because it adds an external dependency (`ffmpeg`) and you may want to re-encode with different settings without re-rendering.

If you see `NotOpenSSLWarning` from `urllib3` on macOS, it's usually because you're using the system Python linked against LibreSSL.
This repo pins `urllib3<2` in requirements to avoid that warning. Best long-term fix is using a Homebrew or python.org Python (OpenSSL).