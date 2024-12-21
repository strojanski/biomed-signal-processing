import os
import requests
from bs4 import BeautifulSoup
import re

def sanitize_filename(filename):
    """Remove or replace invalid characters in the filename."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def sanitize_directory_name(directory):
    """Sanitize the directory name."""
    return re.sub(r'[<>:"/\\|?*]', '_', directory)


def download_file(url, save_dir):
    """Download a file and save it in the specified directory."""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, auth=AUTH)
    if response.status_code == 200:
        raw_filename = url.split("/")[-1]
        filename = sanitize_filename(raw_filename)
        filepath = os.path.join(save_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure parent directories exist
        with open(filepath, "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to access: {url}, status code: {response.status_code}")


def parse_and_download(base_url, save_dir, visited=None):
    """Recursively parse the URL and download .png files."""
    if visited is None:
        visited = set()

    # Skip if already visited
    if base_url in visited:
        return
    visited.add(base_url)

    # Fetch the page content
    response = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"}, auth=AUTH)
    if response.status_code != 200:
        print(f"Failed to access: {base_url}, status code: {response.status_code}")
        return

    # Parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")
    os.makedirs(save_dir, exist_ok=True)

    # Loop through all links
    for link in soup.find_all("a"):
        href = link.get("href")
        if not href or href in ["../", "/"]:
            continue

        # Construct full URL
        full_url = base_url + href if not href.startswith("http") else href

        # If the link is a directory, recurse into it
        if href.endswith("/"):
            new_dir = os.path.join(save_dir, sanitize_directory_name(href.strip("/")))
            parse_and_download(full_url, new_dir, visited)
        else:
            # Only download .png files
            if href.lower().endswith(".png"):
                download_file(full_url, save_dir)


# Example Usage
BASE_URL = "https://lbcsi.fri.uni-lj.si/OBSS/Data/CTMRI/"
DOWNLOAD_DIR = "./data/download"
AUTH = ("bsip", "bsip2016")

parse_and_download(BASE_URL, DOWNLOAD_DIR)
