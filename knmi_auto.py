import os
import re
import time
import requests
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))

BASE = ("https://api.dataplatform.knmi.nl/open-data/v1/datasets/"
        "windlidar_nz_wp_platform_10min/versions/1")

PLATFORMS = ['BSB', 'HKZA', 'HKZB', 'HKN', 'HKWA', 'HKWB']
MIN_DAYS = 20             # skip a platform whose best run is shorter
MAX_DAYS = 150            # download at most this many days per platform
PAGE_PAUSE = 2.5
FILE_PAUSE = 2.0
FILE_TRIES = 3            # passes over files that failed to download

session = requests.Session()


class AuthError(RuntimeError):
    pass


def _wait(r, delay):
    """Seconds to wait before a retry: Retry-After if the server sent one."""
    ra = r.headers.get("Retry-After") if r is not None else None
    if ra and ra.strip().isdigit():
        return int(ra.strip())
    return delay


def get(url, params=None, tries=5):
    delay = 20
    for attempt in range(tries):
        last = attempt == tries - 1
        try:
            r = session.get(url, params=params, timeout=90)
        except requests.RequestException as e:
            if last:
                print(f"      network error ({e}); giving up")
                break
            print(f"      network error ({e}); retrying in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 300)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (401, 403):
            raise AuthError(f"HTTP {r.status_code} from KNMI: the API key is "
                            f"missing, invalid or not allowed for this "
                            f"dataset. {r.text[:160]}")
        if r.status_code == 429 or r.status_code >= 500:
            if last:
                print(f"      HTTP {r.status_code}; giving up after "
                      f"{tries} tries")
                break
            wait = _wait(r, delay)
            print(f"      HTTP {r.status_code}; waiting {wait}s "
                  f"({attempt + 1}/{tries})")
            time.sleep(wait)
            delay = min(delay * 2, 300)
            continue
        print(f"      HTTP {r.status_code}: {r.text[:160]}")
        return None
    return None


def day_of(fn):
    m = re.search(r'_10min_(\d{8})\d{4}_', fn)
    return m.group(1) if m else None


def list_platform(platform):
    """Full listing for one platform, or None if the listing failed part-way
    (a partial list would silently shorten the chosen run)."""
    prefix = f"ZephIR_windlidar_{platform}_10min_"
    names, cursor, page = [], f"ZephIR_windlidar_{platform}_", 0
    while True:
        r = get(f"{BASE}/files",
                params={"maxKeys": "500", "orderBy": "filename",
                        "sorting": "asc", "begin": cursor})
        if r is None:
            print("      listing failed part-way; skipping platform")
            return None
        batch = [f["filename"] for f in r.json().get("files", [])]
        if not batch:
            break
        keep = [n for n in batch if n.startswith(prefix)]
        new = set(keep) - set(names)
        names += keep
        page += 1
        print(f"      page {page}: {len(set(names))} files")
        if len(keep) < len(batch) or not new or batch[-1] == cursor:
            break            # left the platform's prefix, or cursor repeated
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
    part = path + ".part"
    try:
        with requests.get(url, stream=True, timeout=300) as d:
            d.raise_for_status()
            expected = d.headers.get("Content-Length")
            n = 0
            with open(part, "wb") as f:
                for chunk in d.iter_content(chunk_size=8192):
                    f.write(chunk)
                    n += len(chunk)
            if expected is not None and n != int(expected):
                raise IOError(f"got {n} of {expected} bytes")
        os.replace(part, path)          # only complete files get the name
    except Exception as e:
        print(f"      failed {filename}: {e}")
        if os.path.exists(part):
            os.remove(part)
        return False
    return True


def main():
    api_key = os.environ.get("KNMI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set the KNMI_API_KEY environment variable before running this "
            "script. Request a free key at "
            "https://developer.dataplatform.knmi.nl/")
    session.headers.update({"Authorization": api_key})

    print("KNMI ZephIR 10-minute records — automatic fetch")
    print("Leave this running. It paces itself to respect the shared "
          "API rate limit.\n")
    summary, missing = [], {}

    for platform in PLATFORMS:
        print(f"\n=== {platform} " + "=" * 46)
        names = list_platform(platform)
        if names is None:
            summary.append((platform, 0, 0, "listing failed", 0, 0))
            continue
        if not names:
            print(f"   no files found for {platform}")
            summary.append((platform, 0, 0, "none found", 0, 0))
            continue

        start, end, n_days = longest_run(names)
        days_all = sorted({d for d in (day_of(x) for x in names) if d})
        print(f"   {len(names)} files, {days_all[0]} to {days_all[-1]}")
        print(f"   longest unbroken run: {n_days} days "
              f"({start} to {end})")

        if n_days < MIN_DAYS:
            print(f"   too fragmented (under {MIN_DAYS} days) — skipping")
            summary.append((platform, n_days, 0, "skipped: too fragmented",
                            0, 0))
            continue

        n_take = min(n_days, MAX_DAYS)
        if n_days > MAX_DAYS:
            end = (datetime.strptime(start, "%Y%m%d")
                   + timedelta(days=MAX_DAYS - 1)).strftime("%Y%m%d")
            print(f"   capping at {MAX_DAYS} days: {start} to {end}")

        want = sorted(n for n in names
                      if (d := day_of(n)) and start <= d <= end)
        folder = os.path.join(HERE, f"knmi_{platform.lower()}")
        os.makedirs(folder, exist_ok=True)
        mins = len(want) * (FILE_PAUSE + 1) / 60
        print(f"   downloading {len(want)} files -> {folder}/ "
              f"(about {mins:.0f} min)")

        todo = list(want)
        for attempt in range(FILE_TRIES):
            if attempt:
                print(f"   retry pass {attempt}: {len(todo)} file(s)")
            failed = []
            for i, n in enumerate(todo, 1):
                if not download_one(n, folder):
                    failed.append(n)
                if i % 10 == 0 or i == len(todo):
                    print(f"      {i}/{len(todo)}  ({len(failed)} failed)")
                time.sleep(FILE_PAUSE)
            todo = failed
            if not todo:
                break
        if todo:
            missing[platform] = todo

        summary.append((platform, n_days, n_take, f"{start}-{end}",
                        len(want) - len(todo), len(want)))

    print("\n\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for p, n_days, n_take, rng, ok, n_want in summary:
        print(f"  {p:<6} best run {n_days:>4} d, took {n_take:>4} d   "
              f"{rng:<24} {ok:>4}/{n_want} files")
    usable = [s for s in summary if s[4] > 0]
    print(f"\n{len(usable)} platforms downloaded.")
    if missing:
        print("\nFiles still missing after retries:")
        for p, lst in missing.items():
            for n in lst:
                print(f"  {p}: {n}")


if __name__ == '__main__':
    main()
