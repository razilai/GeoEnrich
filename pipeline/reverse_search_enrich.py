"""Stage 3 (OPTIONAL): swap generated images for real web images via reverse search.

Best-effort and fragile by nature (Google scraping can be blocked, requires the
query image to be publicly reachable, and the top-1 hit may be irrelevant). It is
therefore off by default; the reproducible path uses the generated images from
Stage 2. On any per-row failure the row keeps its generated image.

Pipeline per row:
  1. upload the local generated image to a public host  -> public URL
  2. GoogleReverseImageSearch.response(query=description, image_url=URL) -> top hits
  3. resolve the top-1 hit to an actual image URL (og:image or a direct image src)
  4. download it, overwriting row_{idx}.png

Reuses ``GoogleReverseImageSearch`` from ../reverse-img-search (loaded by path
since that directory name isn't a valid module identifier).
"""
from __future__ import annotations

import importlib.util
import os
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _load_ris_class():
    """Import GoogleReverseImageSearch from the sibling reverse-img-search dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "reverse-img-search", "reverse_image_search.py")
    spec = importlib.util.spec_from_file_location("reverse_image_search", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GoogleReverseImageSearch


def upload_to_host(image_path: str) -> str:
    """Upload an image to a public host and return its URL (default: 0x0.st).

    Kept small and swappable — any function ``str -> public_url`` works.
    """
    with open(image_path, "rb") as f:
        resp = requests.post(
            "https://0x0.st", files={"file": f}, headers=_HEADERS, timeout=30
        )
    resp.raise_for_status()
    return resp.text.strip()


def _resolve_image_url(page_url: str) -> str | None:
    """Given a result page URL, find a usable image URL (og:image or first <img>)."""
    if urlparse(page_url).path.lower().endswith(_IMG_EXTS):
        return page_url  # already a direct image link
    resp = requests.get(page_url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return urljoin(page_url, og["content"])
    for img in soup.find_all("img", src=True):
        if img["src"].lower().split("?")[0].endswith(_IMG_EXTS):
            return urljoin(page_url, img["src"])
    return None


def _download(url: str, dest: str) -> bool:
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    if not resp.headers.get("Content-Type", "").startswith("image"):
        return False
    with open(dest, "wb") as f:
        f.write(resp.content)
    return True


def enrich_with_reverse_search(
    csv: str,
    image_folder: str,
    limit: int | None = None,
    uploader=upload_to_host,
) -> None:
    """Replace generated images with top-1 reverse-search results, best-effort."""
    RIS = _load_ris_class()
    searcher = RIS()

    df = pd.read_csv(csv)
    if limit is not None:
        df = df.head(limit)
    df = df.reset_index(drop=True)

    replaced, failed = 0, 0
    for i in range(len(df)):
        fn = f"row_{i:05d}.png"
        local = os.path.join(image_folder, fn)
        if not os.path.exists(local):
            continue
        query = str(df.iloc[i].get("description", "house"))
        try:
            public_url = uploader(local)
            results = searcher.response(query=query, image_url=public_url, max_results=1)
            if isinstance(results, str) or not getattr(results, "results", None):
                failed += 1
                continue
            top_link = results.results[0].get("link")
            img_url = _resolve_image_url(top_link) if top_link else None
            if img_url and _download(img_url, local):
                replaced += 1
            else:
                failed += 1
        except Exception as e:  # keep generated image on any error
            failed += 1
            if failed <= 5:
                print(f"  row {i}: RIS failed ({e}); keeping generated image")

    print(f"✅ reverse-search done: {replaced} replaced, {failed} kept-as-generated")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Optional reverse-image-search enrichment.")
    p.add_argument("--csv", required=True, help="CSV with a 'description' column")
    p.add_argument("--image-folder", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    enrich_with_reverse_search(
        csv=args.csv, image_folder=args.image_folder, limit=args.limit
    )
