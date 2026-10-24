"""Download the Amharic MasakhaNEWS TSV files from the Masakhane GitHub repo.

Usage: python data/download.py
Writes data/raw/{train,dev,test}.tsv (columns: category, headline, text, url).
Check the dataset license/terms in the MasakhaNEWS paper and dataset card before redistributing.
"""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/masakhane-io/masakhane-news/main/data/amh"
OUT = Path(__file__).resolve().parent / "raw"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        url = f"{BASE}/{split}.tsv"
        dest = OUT / f"{split}.tsv"
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    print("done")


if __name__ == "__main__":
    main()
