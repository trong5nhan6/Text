"""CodaBench submission files: predictions.csv + a flat submission.zip.

Format (docs/FORMAT.md, docs/starting_kit/sample_task_*_predictions.csv):
  header `id,label`, one row per id of the input file, ids unchanged, UTF-8,
  label spelled out (Hate / Non-Hate, or Gender / Political / Religion /
  Geo-political / Violence / Others), and the zip holds predictions.csv alone.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def write_submission(ids, probs, labels, out_dir, log=print) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": ids, "label": np.asarray(labels)[np.asarray(probs).argmax(1)]})
    assert len(sub) == len(ids), f"{len(sub)} rows for {len(ids)} ids"
    sub.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")
    np.save(out_dir / "probs.npy", probs)            # kept so the run can be re-blended later
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "predictions.csv", arcname="predictions.csv")
    counts = ", ".join(f"{k} {v}" for k, v in sub.label.value_counts().items())
    log(f"submission -> {out_dir / 'submission.zip'}  ({len(sub)} rows: {counts})")
    return out_dir / "submission.zip"
