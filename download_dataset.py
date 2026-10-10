import os
from pathlib import Path
import gdown

FILE_ID = "1Go3jR73kugTnAdZW5qXkf3yr_WPVLKB6"
OUTPUT_PATH = Path("data/raw/2019-Oct.csv")

def download_dataset():
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        print(f"Dataset already exists: {OUTPUT_PATH}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    result = gdown.download(
        url,
        output=str(OUTPUT_PATH),
        quiet=False
    )

    if not result or not OUTPUT_PATH.exists():
        raise RuntimeError(
            "Download failed. Check Drive sharing permissions "
            "and Google Drive download limits."
        )

    print(f"Dataset downloaded to: {OUTPUT_PATH}")

if __name__ == "__main__":
    download_dataset()