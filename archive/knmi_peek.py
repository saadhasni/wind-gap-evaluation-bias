import requests

API_KEY = ("eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjUzYTg1ZDBhMm"
           "Q5YzRkYzJiYWNlNzQ4NTQ2Zjk4ODExIiwiaCI6Im11cm11cjEyOCJ9")

URL = ("https://api.dataplatform.knmi.nl/open-data/v1/datasets/"
       "windlidar_nz_wp_platform_10min/versions/1/files")

print("Running knmi_peek.py  --  one request only\n")

r = requests.get(URL,
                 headers={"Authorization": API_KEY},
                 params={"maxKeys": "25"},
                 timeout=60)

if r.status_code == 429:
    print("Rate limited (429). The shared anonymous key is busy.")
    print("Wait 30-60 minutes and run this again. One request should get through.")
    raise SystemExit

if r.status_code != 200:
    print(f"HTTP {r.status_code}")
    print(r.text[:500])
    raise SystemExit

files = r.json().get("files", [])
print(f"{len(files)} filenames returned:\n")
for f in files:
    print(f"   {f['filename']}    {f.get('size', 0) / 1024:.0f} KB")

print("\nDone")
