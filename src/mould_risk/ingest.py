"""Stream room temperature/humidity files out of the IDEAL zip and reduce 12 s readings to hourly means."""

import gzip
import io
import os
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from mould_risk.paths import PROCESSED, RAW

ROOM_FILE = re.compile(r"home(\d+)_([a-z]+)(\d+)_sensor(\d+)_room_(temperature|humidity)\.csv(?:\.gz)?$")


def parse_name(member):
    m = ROOM_FILE.search(member.rsplit("/", 1)[-1])
    if not m:
        return None
    home, roomtype, room, sensor, kind = m.groups()
    return {"homeid": int(home), "roomtype": roomtype, "roomid": int(room), "sensorid": int(sensor), "kind": kind}


def read_readings(raw: bytes) -> pd.DataFrame:
    """Two-column timestamp,value CSV, with or without a header row."""
    skip = 1 if raw[:16].lower().startswith(b"timestamp") else 0
    table = pacsv.read_csv(
        io.BytesIO(raw),
        read_options=pacsv.ReadOptions(column_names=["ts", "value"], skip_rows=skip),
        convert_options=pacsv.ConvertOptions(column_types={"ts": pa.timestamp("s"), "value": pa.float64()}),
    )
    return table.to_pandas()


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    hours = df["ts"].dt.floor("h")
    g = df.groupby(hours)["value"]
    return pd.DataFrame({"value": g.mean() / 10.0, "n": g.size()}).rename_axis("ts").reset_index()


def process_member(args):
    zip_path, member = args
    meta = parse_name(member)
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(member)
    if member.endswith(".gz"):
        raw = gzip.decompress(raw)
    hourly = to_hourly(read_readings(raw))
    for k, v in meta.items():
        hourly[k] = v
    return hourly


def room_members(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return [n for n in z.namelist() if parse_name(n)]


def build_room_hourly(zip_path, out_path, workers=None):
    members = room_members(zip_path)
    print(f"{len(members)} room temperature/humidity files", flush=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(process_member, [(zip_path, m) for m in members], chunksize=4), 1):
            parts.append(part)
            if i % 100 == 0:
                print(f"  {i}/{len(members)}", flush=True)
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot_table(
        index=["homeid", "roomid", "roomtype", "ts"], columns="kind", values=["value", "n"], aggfunc="first"
    )
    wide.columns = [f"{kind}_{stat}" if stat == "n" else kind for stat, kind in wide.columns]
    wide = wide.rename(columns={"temperature": "t_in", "humidity": "rh_in"}).reset_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out_path, index=False)
    print(f"wrote {len(wide):,} room-hours to {out_path}")
    return wide


if __name__ == "__main__":
    zip_path = sys.argv[1] if len(sys.argv) > 1 else RAW / "room_and_appliance_sensors.zip"
    build_room_hourly(zip_path, PROCESSED / "room_hourly.parquet")
