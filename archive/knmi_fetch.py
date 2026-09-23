import os
import sys
import time
import json
import requests

# anonymous key from https://developer.dataplatform.knmi.nl/open-data-api
# valid until 1 August 2027
API_KEY = ("eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjUzYTg1ZDBhMm"
           "Q5YzRkYzJiYWNlNzQ4NTQ2Zjk4ODExIiwiaCI6Im11cm11cjEyOCJ9")

BASE = "https://api.dataplatform.knmi.nl/open-data/v1"
DATASET = "windlidar_nz_wp_platform_10min"
VERSION = "1"

PAGE_PAUSE = 2.0          # seconds between listing requests (~30/min)
FILE_PAUSE = 1.5          # seconds between download requests
CACHE = "knmi_filelist.txt"
STATE = "knmi_listing_state.json"

session = requests.Session()
session.headers.update({"Authorization": API_KEY})


def get(url, params=None, tries=6):
    """GET with exponential backoff on 429 / 5xx."""
    delay = 10
    for attempt in range(tries):
        r = session.get(url, params=params, timeout=90)
        if r.status_code == 200:
            return r
        if r.status_code == 429 or r.status_code >= 500:
            print(f"    [{r.status_code}] backing off {delay}s "
                  f"(attempt {attempt + 1}/{tries})")
            time.sleep(delay)
            delay = min(delay * 2, 180)
            continue
        print(f"    [error {r.status_code}] {r.text[:300]}")
        return None
    print("    gave up after repeated rate limiting")
    return None


def peek(n=25):
    """One request. Shows the naming convention."""
    r = get(f"{BASE}/datasets/{DATASET}/versions/{VERSION}/files",
            params={"maxKeys": str(n)})
    if r is None:
        return
    files = r.json().get("files", [])
    print(f"\nFirst {len(files)} filenames:\n")
    for f in files:
        print(f"   {f['filename']}   ({f.get('size', 0) / 1024:.0f} KB)")


def list_all_filenames():
    """Full listing, paced and resumable. Cached to disk."""
    names, token = [], None
    if os.path.exists(CACHE):
        names = [l.strip() for l in open(CACHE, encoding="utf-8") if l.strip()]
        if os.path.exists(STATE):
            token = json.load(open(STATE)).get("token")
        if names and token is None:
            print(f"Using cached listing: {len(names)} filenames "
                  f"(delete {CACHE} to refresh)")
            return names
        print(f"Resuming listing from cache: {len(names)} filenames so far")

    page = 0
    while True:
        params = {"maxKeys": "500"}
        if token:
            params["nextPageToken"] = token
        r = get(f"{BASE}/datasets/{DATASET}/versions/{VERSION}/files",
                params=params)
        if r is None:
            print("Listing interrupted. Rerun to resume from where it stopped.")
            break
        j = r.json()
        batch = [f["filename"] for f in j.get("files", [])]
        names += batch
        token = j.get("nextPageToken")
        page += 1

        with open(CACHE, "w", encoding="utf-8") as fh:
            fh.write("\n".join(names))
        json.dump({"token": token}, open(STATE, "w"))

        print(f"  page {page}: {len(names)} filenames")
        if not token:
            print("  listing complete")
            break
        time.sleep(PAGE_PAUSE)
    return names


def download_one(filename, directory="."):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  already have {filename}")
        return True
    r = get(f"{BASE}/datasets/{DATASET}/versions/{VERSION}"
            f"/files/{filename}/url")
    if r is None:
        return False
    url = r.json().get("temporaryDownloadUrl")
    try:
        with requests.get(url, stream=True, timeout=600) as d:
            d.raise_for_status()
            with open(path, "wb") as f:
                for chunk in d.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"  [failed] {filename}: {e}")
        return False
    print(f"  saved {filename}  ({os.path.getsize(path) / 1024:.0f} KB)")
    return True


def main():
    # ------------------------------------------------------------------
    MODE = 'peek'          # 'peek' | 'survey' | 'sample' | 'bulk'
    PLATFORM = 'BSB'       # BSA BSB HKZA HKZB HKN HKWA HKWB
    START = '20210101'     # bulk only, YYYYMMDD, inclusive
    END = '20210301'       # bulk only, YYYYMMDD, inclusive
    # ------------------------------------------------------------------

    if MODE == 'peek':
        peek()
        print("\nSend these filenames to the chat. One request used.")
        return

    names = list_all_filenames()
    if not names:
        return
    print(f"\nTotal filenames held: {len(names)}")
    print("\nFirst 5 / last 5:")
    for n in names[:5]:
        print("   ", n)
    print("    ...")
    for n in names[-5:]:
        print("   ", n)

    codes = ['BSA', 'BSB', 'HKZA', 'HKZB', 'HKN', 'HKWA', 'HKWB']
    matches = {}
    print("\nFiles per platform:")
    for c in codes:
        # match the code as a delimited token first, so HKN does not also
        # sweep up HKNxx; fall back to a loose substring match if that fails
        hits = [n for n in names
                if c.lower() in n.lower().replace('.', '_').split('_')]
        if not hits:
            hits = [n for n in names if c.lower() in n.lower()]
        matches[c] = hits
        print(f"   {c:<5} {len(hits):>6}")

    if MODE == 'survey':
        print("\nSurvey done. Set MODE = 'sample' and choose PLATFORM next.")
        return

    sel = sorted(matches.get(PLATFORM, []))
    if not sel:
        print(f"\nNothing matched '{PLATFORM}'. Compare the codes above "
              f"against the real filenames printed earlier.")
        return

    if MODE == 'sample':
        print(f"\nDownloading one {PLATFORM} file "
              f"(from the middle of the record) ...")
        download_one(sel[len(sel) // 2])
        print("\nUpload that file to the chat before downloading any more.")
        return

    if MODE == 'bulk':
        def in_range(fn):
            toks = fn.replace('.', '_').split('_')
            return any(t.isdigit() and len(t) == 8 and START <= t <= END
                       for t in toks)

        want = [n for n in sel if in_range(n)]
        folder = f"knmi_{PLATFORM.lower()}"
        print(f"\n{len(want)} {PLATFORM} files in {START}-{END} -> {folder}/")
        if not want:
            print("Nothing in range. Check the date tokens in the filenames.")
            return
        est = len(want) * (FILE_PAUSE + 1) / 60
        print(f"Estimated time: about {est:.0f} minutes. Leave it running.\n")
        ok = 0
        for i, n in enumerate(want, 1):
            if download_one(n, folder):
                ok += 1
            if i % 20 == 0:
                print(f"  ... {i}/{len(want)}")
            time.sleep(FILE_PAUSE)
        print(f"\nDone: {ok}/{len(want)} downloaded into {folder}/")


if __name__ == '__main__':
    main()