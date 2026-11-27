"""Re-run build_report.run_study for Florence_victim only, against the repo's rows_scored.csv.

Usage (from the VLAttack folder):
    python 重算工具/rerun_florence_routing.py 實驗 <output folder>
Then compare <output folder>/Florence_victim with
實驗/資料/分流分析_C1到C7/Florence_victim.
Used on 2026-11-27 to regenerate that folder after an exp3 sample was replaced.

- Uses the unmodified build_report.py from the repo (imported, not copied).
- rows: 實驗/資料/Florence/florence_victim/<exp>/florence2/vqa_florence2/rows_scored.csv
- Images are not in the repo, so copy_images is replaced by a stub that returns the file names the
  original would have copied (keeps the REPORT.md links in the same format) without touching disk.
- Output goes to argv[2] (a scratch folder), never into the repo.
"""
import sys
from pathlib import Path

exp_root = Path(sys.argv[1])  # .../VLAttack/實驗
out_dir = Path(sys.argv[2])
script_dir = exp_root / "程式" / "4_報告產生" / "stage_routing_deep_dive"
tab = exp_root / "資料" / "Florence"
sys.path.insert(0, str(script_dir))

import build_report as br  # noqa: E402


def stub_copy_images(row, vis_dir, dest):
    qid = row["image_id"]
    return [f"{qid}_1_ori.png", f"{qid}_2_atk.png"] if int(row["label"]) == 1 else [f"{qid}_1_ori.png"]


br.copy_images = stub_copy_images
br.OUT = out_dir
meta = dict(br.STUDIES["Florence_victim"])
meta["rows"] = lambda exp: tab / "florence_victim" / exp / "florence2" / "vqa_florence2" / "rows_scored.csv"
meta["visual"] = {e: out_dir / "_no_images" for e in ("exp1", "exp2", "exp3")}
print(br.run_study("Florence_victim", meta))
