"""Load the Wandering Eyes fixations and cut them into labelled 30 s windows (Part A)."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "wandering_eyes" / "Data"
WINDOW_MS = 30_000

# Probe onsets in ms, from the paper; they match the processed data.csv.
PROBES = {
    "Education": [168_000, 388_000, 691_000, 1_043_000],
    "Genetics": [199_000, 648_000, 901_000, 1_275_000],
}
# AOIs copied from the authors' notebook (x1, x2, y1, y2).
AOI = {
    "Education": {"slides": (167.5, 944.6, 212.1, 807.8), "teacher": (979.1, 1266.9, 383.7, 704.1)},
    "Genetics": {"slides": (465, 1217, 200, 800), "teacher": (67, 440.9, 200, 800)},
}
FILES = {
    ("probe", "Education"): "education", ("probe", "Genetics"): "genetics",
    ("self", "Education"): "education", ("self", "Genetics"): "genetics",
}


def _inside(x, y, box):
    x1, x2, y1, y2 = box
    return (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)


def load_fixations(cond: str, video: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"full_fix_data_{cond}_{FILES[(cond, video)]}.csv")
    df = df.rename(columns={"SubjectName": "subject", "StartTime": "t", "Length": "dur", "PosX": "x", "PosY": "y"})
    df = df[(df.dur >= 80) & (df.dur <= 2000)].copy()
    box = AOI[video]
    aoi = np.where(_inside(df.x, df.y, box["slides"]), "slides",
                   np.where(_inside(df.x, df.y, box["teacher"]), "teacher",
                            np.where((df.x == 0) & (df.y == 0), "miss",
                                     np.where((df.x > 0) & (df.x < 1280) & (df.y > 152) & (df.y < 872), "ws_in", "out"))))
    df["aoi"] = aoi
    df["video"], df["cond"] = video, cond
    df["pid"] = f"{cond}_{video}_" + df.subject
    return df.sort_values(["pid", "t"]).reset_index(drop=True)


def self_reports(video: str) -> pd.DataFrame:
    k = pd.read_csv(RAW / f"key_self_{FILES[('self', video)]}.csv").rename(columns={"SubjectName": "subject", "kd": "t"})
    k = k.sort_values(["subject", "t"])
    # authors' rule: of two presses within 5 s, drop the first
    nxt = k.groupby("subject").t.shift(-1)
    k = k[~((nxt - k.t) < 5000)]
    k["pid"] = f"self_{video}_" + k.subject
    return k.reset_index(drop=True)


def probe_labels(video: str) -> pd.DataFrame:
    k = pd.read_csv(RAW / f"key_probe_{FILES[('probe', video)]}.csv").rename(columns={"SubjectName": "subject"})
    k["t"] = k.segment.map(dict(enumerate(PROBES[video])))
    k["y"] = (k.key == "y").astype(int)
    k["pid"] = f"probe_{video}_" + k.subject
    return k[["pid", "subject", "segment", "t", "y"]]


def window(fix: pd.DataFrame, end: int, length: int = WINDOW_MS) -> pd.DataFrame:
    return fix[(fix.t >= end - length) & (fix.t < end)]


def self_windows(video: str, lag_ms: int = 0, length: int = WINDOW_MS) -> list[dict]:
    """Positives end `lag_ms` before each report; negatives on a 30 s grid away from reports."""
    fix = load_fixations("self", video)
    rep = self_reports(video)
    out = []
    for pid, f in fix.groupby("pid"):
        r = rep[rep.pid == pid].t.to_numpy()
        for t in r:
            out.append({"pid": pid, "video": video, "end": int(t - lag_ms), "y": 1, "fix": window(f, t - lag_ms, length)})
        last = f.t.max()
        for end in range(length, int(last) + 1, WINDOW_MS):
            # not within 30 s before nor 15 s after any report
            if len(r) and np.any((r - end >= 0) & (r - end < WINDOW_MS)) | np.any((end - r > 0) & (end - r <= 15_000 + length)):
                continue
            out.append({"pid": pid, "video": video, "end": end, "y": 0, "fix": window(f, end, length)})
    return out


def probe_windows(video: str, length: int = WINDOW_MS) -> list[dict]:
    fix = load_fixations("probe", video)
    lab = probe_labels(video)
    out = []
    for row in lab.itertuples():
        f = fix[fix.pid == row.pid]
        out.append({"pid": row.pid, "video": video, "end": int(row.t), "y": int(row.y), "fix": window(f, row.t, length)})
    return out


def grid_windows(cond: str, video: str, length: int = WINDOW_MS) -> list[dict]:
    """All consecutive 30 s windows of the lecture, unlabelled (for H6 and H7)."""
    fix = load_fixations(cond, video)
    out = []
    for pid, f in fix.groupby("pid"):
        for end in range(length, int(f.t.max()) + 1, WINDOW_MS):
            out.append({"pid": pid, "video": video, "end": end, "y": -1, "fix": window(f, end, length)})
    return out
