"""Build feature tables for A1 (probe), A2 (self, lags 0-30 s) and the full-lecture grids."""
import json
import pickle
from pathlib import Path

from data import grid_windows, probe_windows, self_windows
from features import apply_exclusions, sequence, table

OUT = Path(__file__).resolve().parents[1] / "data" / "derived"
OUT.mkdir(parents=True, exist_ok=True)
VIDEOS = ["Education", "Genetics"]
log = {}

a1 = [w for v in VIDEOS for w in probe_windows(v)]
t, log["A1"] = apply_exclusions(table(a1))
t.to_csv(OUT / "a1_probe.csv", index=False)

for lag in [0, 5, 10, 20, 30]:
    ws = [w for v in VIDEOS for w in self_windows(v, lag_ms=lag * 1000)]
    t, log[f"A2_lag{lag}"] = apply_exclusions(table(ws))
    t.to_csv(OUT / f"a2_self_lag{lag}.csv", index=False)
    if lag == 0:
        keep = set(zip(t.pid, t.end, t.y))
        seqs = [(w["pid"], w["video"], w["end"], w["y"], sequence(w["fix"])) for w in ws if (w["pid"], w["end"], w["y"]) in keep]
        pickle.dump(seqs, open(OUT / "a2_sequences.pkl", "wb"))

for cond in ["self", "probe"]:
    g = [w for v in VIDEOS for w in grid_windows(cond, v)]
    t = table(g)
    t.to_csv(OUT / f"grid_{cond}.csv", index=False)

json.dump(log, open(OUT / "exclusions.json", "w"), indent=1)
print(json.dumps(log, indent=1))
