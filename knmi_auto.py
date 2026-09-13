import os
import re
import time
import requests
from datetime import datetime, timedelta

API_KEY = os.environ.get("KNMI_API_KEY")
if not API_KEY:
    raise SystemExit(
        "Set the KNMI_API_KEY environment variable before running this script. "
        "Request a free key at https://developer.dataplatform.knmi.nl/"
    )

BASE = ("https://api.dataplatform.knmi.nl/open-data/v1/datasets/"
        "windlidar_nz_wp_platform_10min/versions/1")

PLATFORMS = ['BSB', 'HKZA', 'HKZB', 'HKN', 'HKWA', 'HKWB']
MIN_DAYS = 20             # skip a platform whose best run is shorter
MAX_DAYS = 150            # download at most this many days per platform
PAGE_PAUSE = 2.5
FILE_PAUSE = 2.0

session = requests.Session()
session.headers.update({"Authorization": API_KEY})


def get(url, params=None, tries=5):
    delay = 20
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=90)
        except requests.RequestException as e:
            print(f"      network error ({e}); retrying in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 300)
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 429 or r.status_code >= 500:
            print(f"      rate limited; waiting {delay}s "
                  f"({attempt + 1}/{tries})")
            time.sleep(delay)
            delay = min(delay * 2, 300)
            continue
        print(f"      HTTP {r.status_code}: {r.text[:160]}")
        return None
    return None


def day_of(fn):
    m = re.search(r'_10min_(\d{8})\d{4}_', fn)
    return m.group(1) if m else None


def list_platform(platform):
    prefix = f"ZephIR_windlidar_{platform}_10min_"
    names, cursor, page = [], f"ZephIR_windlidar_{platform}_", 0
    while True:
        r = get(f"{BASE}/files",
                params={"maxKeys": "500", "orderBy": "filename",
                        "sorting": "asc", "begin": cursor})
        if r is None:
            print("      listing stopped early")
            break
        batch = [f["filename"] for f in r.json().get("files", [])]
        if not batch:
            break
        keep = [n for n in batch if n.startswith(prefix)]
        names += keep
        page += 1
        print(f"      page {page}: {len(names)} files")
        if len(keep) < len(batch):
            break
        cursor = batch[-1]
        time.sleep(PAGE_PAUSE)
    return sorted(set(names))


def longest_run(names):
    """Return (start, end, length) of the longest unbroken run of days."""
    days = sorted({d for d in (day_of(n) for n in names) if d})
    if not days:
        return None, None, 0
    have = set(days)
    d0 = datetime.strptime(days[0], "%Y%m%d")
    d1 = datetime.strptime(days[-1], "%Y%m%d")
    best_len = run_len = 0
    best_start = run_start = None
    day = d0
    while day <= d1:
        s = day.strftime("%Y%m%d")
        if s in have:
            if run_len == 0:
                run_start = s
            run_len += 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0
        day += timedelta(days=1)
    if not best_start:
        return None, None, 0
    end = (datetime.strptime(best_start, "%Y%m%d")
           + timedelta(days=best_len - 1)).strftime("%Y%m%d")
    return best_start, end, best_len


def download_one(filename, folder):
    path = os.path.join(folder, filename)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return True
    r = get(f"{BASE}/files/{filename}/url")
    if r is None:
        return False
    url = r.json().get("temporaryDownloadUrl")
    try:
        with requests.get(url, stream=True, timeout=300) as d:
            d.raise_for_status()
            with open(path, "wb") as f:
                for chunk in d.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"      failed {filename}: {e}")
        if os.path.exists(path):
            os.remove(path)
        return False
    return True


def main():
    print("KNMI ZephIR 10-minute records — automatic fetch")
    print("Leave this running. It paces itself to respect the shared "
          "API rate limit.\n")
    summary = []

    for platform in PLATFORMS:
        print(f"\n=== {platform} " + "=" * 46)
        names = list_platform(platform)
        if not names:
            print(f"   no files found for {platform}")
            summary.append((platform, 0, "none found", 0))
            continue

        start, end, n_days = longest_run(names)
        days_all = sorted({d for d in (day_of(x) for x in names) if d})
        print(f"   {len(names)} files, {days_all[0]} to {days_all[-1]}")
        print(f"   longest unbroken run: {n_days} days "
              f"({start} to {end})")

        if n_days < MIN_DAYS:
            print(f"   too fragmented (under {MIN_DAYS} days) — skipping")
            summary.append((platform, n_days, "skipped: too fragmented", 0))
            continue

        if n_days > MAX_DAYS:
            end = (datetime.strptime(start, "%Y%m%d")
                   + timedelta(days=MAX_DAYS - 1)).strftime("%Y%m%d")
            print(f"   capping at {MAX_DAYS} days: {start} to {end}")

        want = sorted(n for n in names
                      if (d := day_of(n)) and start <= d <= end)
        folder = f"knmi_{platform.lower()}"
        os.makedirs(folder, exist_ok=True)
        mins = len(want) * (FILE_PAUSE + 1) / 60
        print(f"   downloading {len(want)} files -> {folder}/ "
              f"(about {mins:.0f} min)")

        ok = 0
        for i, n in enumerate(want, 1):
            if download_one(n, folder):
                ok += 1
            if i % 10 == 0 or i == len(want):
                print(f"      {i}/{len(want)}  ({ok} on disk)")
            time.sleep(FILE_PAUSE)

        summary.append((platform, n_days, f"{start}-{end}", ok))

    print("\n\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for p, n_days, rng, ok in summary:
        print(f"  {p:<6} best run {n_days:>4} days   {rng:<24} "
              f"{ok:>4} files")
    print("\nSummary")
    usable = [s for s in summary if s[3] > 0]
    if usable:
        print(f"\n{len(usable)} platforms downloaded.")


if __name__ == '__main__':
    main()
