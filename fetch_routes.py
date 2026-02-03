import io
import os
import zipfile
import requests

GTFS_URL = "https://infopalvelut.storage.hsldev.com/gtfs/hsl.zip"
OUTPUT_FILE = "routes.txt"


def fetch_routes_txt(output_path: str = OUTPUT_FILE) -> None:
    """
    Download the HSL GTFS package and extract only routes.txt.
    
    The zip contains an 'hsl' folder with all GTFS files.
    We only extract routes.txt and discard everything else.
    """
    print(f"Downloading HSL GTFS package from {GTFS_URL}...")
    
    response = requests.get(GTFS_URL, stream=True)
    response.raise_for_status()
    
    # Get total size for progress indication
    total_size = int(response.headers.get('content-length', 0))
    if total_size:
        print(f"Package size: {total_size / (1024 * 1024):.1f} MB")
    
    # Download to memory
    zip_data = io.BytesIO()
    downloaded = 0
    for chunk in response.iter_content(chunk_size=8192):
        zip_data.write(chunk)
        downloaded += len(chunk)
        if total_size:
            progress = downloaded / total_size * 100
            print(f"\rDownloading: {progress:.1f}%", end="", flush=True)
    
    print("\nExtracting routes.txt...")
    
    # Extract only routes.txt from the zip
    zip_data.seek(0)
    with zipfile.ZipFile(zip_data, 'r') as zf:
        # Find routes.txt (should be at hsl/routes.txt)
        routes_path = None
        for name in zf.namelist():
            if name.endswith('routes.txt'):
                routes_path = name
                break
        
        if routes_path is None:
            raise FileNotFoundError("routes.txt not found in the GTFS package")
        
        # Extract routes.txt content and write to output
        routes_content = zf.read(routes_path)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(routes_content)
    
    print(f"Successfully saved {output_path}")


if __name__ == "__main__":
    fetch_routes_txt()
