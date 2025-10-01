"""
Download one session lap telemetry with FastF1 and save CSV.
Edit YEAR/EVENT/SESSION/DRIVER below as needed.
"""
import fastf1 as ff1
import os
import pandas as pd

# ---------- user changeable ----------
YEAR = 2023
EVENT = "Bahrain"        # e.g. "Bahrain", "Monaco", or full name
SESSION = "Q"            # 'FP1','FP2','FP3','Q','R'
DRIVER = "VER"           # 3-letter driver code e.g. 'VER','LEC','HAM'
CACHE_DIR = "fastf1_cache"
OUT_DIR = "data"
# ------------------------------------

# enable local cache (speeds repeated runs)
ff1.Cache.enable_cache(CACHE_DIR)

print(f"Loading session {EVENT} {YEAR} {SESSION} ...")
# get the session (fastf1 accepts year + event name or index)
session = ff1.get_session(YEAR, EVENT, SESSION)

# load telemetry + laps (may take a minute; large files cached)
session.load(telemetry=True, laps=True)

# pick laps of driver and the fastest lap
laps = session.laps.pick_driver(DRIVER)
if laps.empty:
    raise SystemExit(f"No laps found for driver {DRIVER} in this session.")
fastest = laps.pick_fastest()

# get telemetry for that lap and add distance (useful for plotting)
telemetry = fastest.get_telemetry().add_distance()

# keep useful columns; you can expand this list
cols = [c for c in ["Distance", "Speed", "Throttle", "Brake", "nGear", "RPM", "Time"] if c in telemetry.columns]
df = telemetry[cols]

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"telemetry_{DRIVER}_{EVENT}_{SESSION}_{YEAR}.csv")
df.to_csv(out_path, index=False)
print("Saved telemetry to", out_path)

# quick head print
print(df.head().to_string())
