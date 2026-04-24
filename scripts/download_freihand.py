#!/usr/bin/env python3
"""Download and extract FreiHand dataset."""
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

URLS = {
    "FreiHAND_pub_v2.zip": "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip",
    "FreiHAND_pub_v2_eval.zip": "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip",
}


def download(url, dest):
    if dest.exists():
        print(f"  Exists: {dest}")
        return
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(
        url, str(dest),
        reporthook=lambda b, bs, ts: print(f"\r  {b*bs/1e6:.0f}/{ts/1e6:.0f} MB", end=""),
    )
    print()


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/freihand")
    data_dir.mkdir(parents=True, exist_ok=True)

    for fname, url in URLS.items():
        dest = data_dir / fname
        download(url, dest)
        if dest.exists() and not (data_dir / "training").exists():
            print(f"  Extracting {fname} ...")
            with zipfile.ZipFile(dest, "r") as z:
                z.extractall(data_dir)

    imgs = list((data_dir / "training" / "rgb").glob("*.jpg")) if (data_dir / "training" / "rgb").exists() else []
    print(f"\nFreiHand ready: {len(imgs)} training images in {data_dir}")


if __name__ == "__main__":
    main()
