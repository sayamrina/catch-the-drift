"""Copy what the demo and the project page need (Education lecture, self-caught group only) into app/data and docs."""
import json
import shutil
from pathlib import Path

import pandas as pd

from data import load_fixations, self_reports

ROOT = Path(__file__).resolve().parents[1]
RES, APP, DOCS = ROOT / "results", ROOT / "app" / "data", ROOT / "docs"
APP.mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data" / "raw" / "wandering_eyes" / "Data"

a = json.load(open(RES / "part_a.json"))
x = json.load(open(RES / "exploratory.json"))
gs = pd.read_csv(RES / "grid_self_scored.csv")
gs = gs[gs.video == "Education"][["pid", "end", "p"]].round({"p": 4})
gs.to_csv(APP / "scores.csv", index=False)
rep = self_reports("Education")
rep[rep.pid.isin(gs.pid)][["pid", "t"]].to_csv(APP / "reports.csv", index=False)
fx = load_fixations("self", "Education")
fx[fx.pid.isin(gs.pid)][["pid", "t", "dur", "x", "y"]].round(1).to_csv(APP / "fixations.csv", index=False)
shutil.copy(RAW / "education_snapshot.jpg", APP / "education_snapshot.jpg")
shutil.copy(RES / "catchup_questions.json", APP / "catchup_questions.json")
meta = {"threshold": a["H7"]["threshold"],
        "results": {"participants": a["exclusions"]["A2_lag0"]["participants_after"], "auc": a["H2"]["logreg"]["auc_roc"],
                    "hit_triggered": a["H7"]["hit_share_triggered"], "hit_fixed": a["H7"]["hit_share_fixed"],
                    "hit_random": a["H7"]["hit_share_random"], "p_vs_fixed": a["H7"]["wilcoxon_p"],
                    "p_vs_random": x["triggered_vs_random_wilcoxon_p"]}}
json.dump(meta, open(APP / "meta.json", "w"), indent=1)
print(meta)
