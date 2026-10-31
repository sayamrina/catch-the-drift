"""The planned window features, and the fixation sequences for the GRU."""
import numpy as np
import pandas as pd

from data import WINDOW_MS

FEATURES = [
    "fix_rate", "dur_mean", "dur_median", "dur_sd", "long_prop", "teacher_prop", "out_prop",
    "slide_dispersion", "step_mean", "step_sd", "horizontal_prop", "coverage",
]
AOIS = ["slides", "teacher", "ws_in", "out", "miss"]


def window_features(f: pd.DataFrame, length: int = WINDOW_MS) -> dict:
    n = len(f)
    if n < 2:
        return {k: np.nan for k in FEATURES} | {"coverage": f.dur.sum() / length if n else 0.0}
    st = (f.aoi == "slides").sum()
    te = (f.aoi == "teacher").sum()
    s = f[f.aoi == "slides"]
    disp = np.sqrt(((s.x - s.x.mean()) ** 2 + (s.y - s.y.mean()) ** 2).mean()) if len(s) > 1 else np.nan
    dx, dy = np.diff(f.x.to_numpy()), np.diff(f.y.to_numpy())
    step = np.hypot(dx, dy)
    return {
        "fix_rate": n / (length / 1000),
        "dur_mean": f.dur.mean(),
        "dur_median": f.dur.median(),
        "dur_sd": f.dur.std(),
        "long_prop": (f.dur >= 500).mean(),
        "teacher_prop": te / (st + te) if st + te else np.nan,
        "out_prop": f.aoi.isin(["out", "miss"]).mean(),
        "slide_dispersion": disp,
        "step_mean": step.mean(),
        "step_sd": step.std(),
        "horizontal_prop": (np.abs(dx) > np.abs(dy)).mean(),
        "coverage": f.dur.sum() / length,
    }


def table(windows: list[dict], length: int = WINDOW_MS) -> pd.DataFrame:
    rows = [{"pid": w["pid"], "video": w["video"], "end": w["end"], "y": w["y"]} | window_features(w["fix"], length)
            for w in windows]
    return pd.DataFrame(rows)


def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Planned: drop windows with <50% fixation coverage, then participants with <2 windows."""
    n0, p0 = len(df), df.pid.nunique()
    df = df[df.coverage >= 0.5]
    keep = df.pid.value_counts()
    df = df[df.pid.isin(keep[keep >= 2].index)].reset_index(drop=True)
    return df, {"windows_before": n0, "windows_after": len(df), "participants_before": p0,
                "participants_after": df.pid.nunique()}


def sequence(f: pd.DataFrame, max_len: int = 120) -> np.ndarray:
    """Per fixation: duration (s), x/1280, y/1024, distance to previous fixation /1000, AOI one-hot."""
    x = f.x.to_numpy() / 1280
    y = f.y.to_numpy() / 1024
    step = np.r_[0.0, np.hypot(np.diff(f.x.to_numpy()), np.diff(f.y.to_numpy())) / 1000]
    onehot = np.stack([(f.aoi == a).to_numpy() for a in AOIS], axis=1).astype(float)
    seq = np.column_stack([f.dur.to_numpy() / 1000, x, y, step, onehot])[-max_len:]
    return seq.astype(np.float32)
