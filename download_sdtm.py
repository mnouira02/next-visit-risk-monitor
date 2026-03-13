import os
import requests

# GitHub API Endpoint for the directory
API_URL = "https://api.github.com/repos/phuse-org/TestDataFactory/contents/Updated/TDF_SDTM"
DATA_DIR = "data"

def download_all_sdtm():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Fetching file list from PhUSE repository...")
    try:
        # Get list of files via API
        r = requests.get(API_URL)
        r.raise_for_status()
        files = r.json()
    except Exception as e:
        print(f"Error fetching file list: {e}")
        return

    # Filter for .xpt files
    xpt_files = [f for f in files if f['name'].endswith('.xpt')]
    print(f"Found {len(xpt_files)} SDTM datasets.")

    for file_info in xpt_files:
        name = file_info['name']
        download_url = file_info['download_url']
        dest_path = os.path.join(DATA_DIR, name)

        # Skip if already exists to save bandwidth
        if os.path.exists(dest_path):
            print(f"Skipping {name} (already exists)")
            continue

        print(f"Downloading {name}...")
        file_resp = requests.get(download_url, stream=True)
        if file_resp.status_code == 200:
            with open(dest_path, 'wb') as f:
                for chunk in file_resp.iter_content(4096):
                    f.write(chunk)
        else:
            print(f"Failed to download {name}")

    print("\nAll datasets downloaded to /data folder.")

if __name__ == "__main__":
    download_all_sdtm()