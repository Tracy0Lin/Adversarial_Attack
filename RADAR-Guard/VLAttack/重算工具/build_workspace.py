"""Build a run-ready workspace that mirrors the original machine's folder layout.

The experiment scripts find their data relative to their own location, the way they were
laid out on the original machine (BLIP_attack/..., BLIP_attack/output/..., BLIP_attack/output/VQA/...).
This delivery keeps code and data in separate folders, so the scripts cannot find their inputs
when run from the repo. This tool copies code and data into a separate folder, back in the
original layout. The repo itself is not modified.

Usage (from the VLAttack folder):
    python 重算工具/build_workspace.py <workspace folder>

Then, for example (CPU only, uses the stored caption features and probe results):
    cd <workspace>/BLIP_attack/output/multi_captioner
    python analyze.py --exp exp1 exp2 exp3 --captioner blip --output-tag sigma08_fixed \
        --fix-theta-star 0.8 --fix-theta-det 0.8

Script placement comes from 溯源/SCRIPT_ORIGIN_MAP.tsv (the original path of each script),
used only when the original file name matches; otherwise a folder-level rule is applied.
Data placement follows the table DATA_RULES below. See WORKSPACE_README.txt in the output.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path


def _long(p: Path) -> str:
    """Windows paths over 260 characters need the \\\\?\\ prefix."""
    s = str(p.resolve())
    return "\\\\?\\" + s if os.name == "nt" and len(s) > 240 and not s.startswith("\\\\?\\") else s


def _copy(src: Path, dst: Path) -> None:
    os.makedirs(_long(dst.parent), exist_ok=True)
    shutil.copy2(_long(src), _long(dst))

REPO = Path(__file__).resolve().parents[1]  # RADAR-Guard/VLAttack
CODE = REPO / "實驗" / "程式"
DATA = REPO / "實驗" / "資料"
VQA = Path("BLIP_attack") / "output" / "VQA"

SKIP_CODE = (
    "_備份_勿執行/",                              # backups must not run
    "2_訓練/",                                    # repo-native script, runs directly from the repo
    "1_攻擊與資料建立/cleverhans/cleverhans_v3.1.0/",  # old upstream examples/docs; the attack uses cleverhans/cleverhans/torch
)

# Folder-level fallback for scripts whose original path is unknown or ambiguous.
CODE_RULES = [
    ("1_攻擊與資料建立/cleverhans/", Path("cleverhans")),
    ("1_攻擊與資料建立/", Path("BLIP_attack")),
    ("3_評估/", Path("BLIP_attack") / "output" / "multi_captioner"),
    ("4_報告產生/stage_routing_deep_dive/", VQA / "stage_routing_deep_dive"),
    ("4_報告產生/實驗二_cos_vs_角距_圖表/", Path("BLIP_attack") / "output" / "實驗二_cos_vs_角距_圖表"),
    ("4_報告產生/", Path("BLIP_attack") / "output" / "multi_captioner"),
    ("5_題目難度/", VQA / "question_difficulty"),
    ("Florence/攻擊/", Path("BLIP_attack") / "output"),
    ("Florence/評估/", Path("BLIP_attack") / "output"),
]

# Data: (repo folder under 實驗/資料, workspace folder). Applied to every file below the repo folder.
DATA_RULES = [
    ("BLIP/舊流程/steps100_alpha0.002_run1000", VQA / "steps100_alpha0.002_run1000"),
    ("BLIP/舊流程/exp1_mixed1000", VQA / "exp1_mixed1000"),
    ("BLIP/舊流程/exp2_success1000", VQA / "exp2_success1000"),
    ("BLIP/Stage1正式_11-22", VQA / "rotation_cos_vs_ang"),
    ("BLIP/攻擊參數搜尋", VQA / "attack_param_search"),
    ("Florence/florence_vlattack", VQA / "florence_vlattack"),
    ("Florence/florence_victim", VQA / "florence_victim"),
    ("分流分析_C1到C7", VQA / "stage_routing_deep_dive"),
    ("題目難度", VQA / "question_difficulty"),
    ("補充資料表", VQA / "data_supplement"),
    ("輸入/annotations", Path("BLIP_attack") / "annotation"),
]


def pick_original(row: dict, repo_name: str) -> Path | None:
    """Original location relative to the original VLAttack root, if it is unambiguous."""
    for marker in ("\\VLAttack_zip\\", "\\program\\VLAttack\\"):
        for o in (row.get("original_paths") or "").split(";"):
            o = o.split(" [")[0]
            if marker not in o:
                continue
            rel = o.split(marker, 1)[1]
            parts = rel.split("\\")
            if parts[0] not in ("BLIP_attack", "cleverhans") or parts[-1] != repo_name:
                continue
            return Path(*parts)
    return None


def place_code(ws: Path, log: list) -> None:
    origin = {r["repo_path"].replace("RADAR-Guard/VLAttack/", ""): r
              for r in csv.DictReader(open(REPO / "溯源" / "SCRIPT_ORIGIN_MAP.tsv", encoding="utf-8"), delimiter="\t")}
    placed: dict[Path, Path] = {}
    for src in sorted(CODE.rglob("*")):
        if not src.is_file() or "__pycache__" in src.parts:
            continue
        rel = src.relative_to(CODE).as_posix()
        if rel.startswith(SKIP_CODE) or src.name == "README.md" and "實驗/程式/" + rel not in origin:
            continue
        row = origin.get("實驗/程式/" + rel)
        target, how = (pick_original(row, src.name), "原始位置") if row else (None, "")
        if target is None:
            for prefix, base in CODE_RULES:
                if rel.startswith(prefix):
                    target, how = base / rel[len(prefix):], "資料夾規則"
                    break
        if target is None:
            log.append(("略過（無對應位置）", rel, ""))
            continue
        if target in placed:
            if placed[target].read_bytes().replace(b"\r\n", b"\n") != src.read_bytes().replace(b"\r\n", b"\n"):
                log.append(("衝突，保留先放的", rel, str(target)))
            continue
        dst = ws / target
        _copy(src, dst)
        placed[target] = src
        log.append((how, rel, target.as_posix()))


def place_data(ws: Path, log: list) -> None:
    mc = DATA / "BLIP" / "主流程_multi_captioner"
    for src in sorted(mc.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(mc)
        head = rel.parts[0]
        if head.endswith("_blip") and head[:-5] in ("exp1", "exp2", "exp3"):
            exp = head[:-5]
            # probe_gaussian.csv is captioner-independent and sits one level up (common.exp_dir(exp))
            target = (VQA / "multi_captioner" / exp / rel.name if rel.name == "probe_gaussian.csv" and len(rel.parts) == 2
                      else VQA / "multi_captioner" / exp / "blip" / Path(*rel.parts[1:]))
        else:
            target = VQA / "multi_captioner" / rel
        copy(src, ws / target, log, "資料")
    for repo_rel, base in DATA_RULES:
        root = DATA / repo_rel
        for src in sorted(root.rglob("*")):
            if src.is_file():
                copy(src, ws / base / src.relative_to(root), log, "資料")
    # attack logs and manifests of the Florence studies were collected flat in 實驗清單/
    for src in sorted((DATA / "實驗清單").glob("florence_*")):
        for study in ("florence_vlattack", "florence_victim"):
            if src.name.startswith(study + "_"):
                rest = src.name[len(study) + 1:]
                for exp_dir in ("exp1_mixed1000", "exp2_success1000", "exp3_success1000"):
                    if rest.startswith(exp_dir + "_"):
                        copy(src, ws / VQA / study / exp_dir / rest[len(exp_dir) + 1:], log, "資料（實驗清單）")


def derive_blip_exp2_log(ws: Path, log: list) -> None:
    """BLIP exp2 attack log is not in the repo. build_success_dataset.py picked exp2 from the attack log
    in log order; its dataset_manifest.csv keeps that order, and every entry is in the 2,298-question
    snapshot. Rebuild the log as the snapshot entries for the manifest's question ids, in manifest order."""
    target = ws / VQA / "exp2_success1000" / "attack_dual_defense_log.json"
    if target.exists():
        return
    snap = json.loads((DATA / "BLIP" / "主流程_multi_captioner" / "attack_log_snapshot_2298.json").read_text(encoding="utf-8"))
    by_id = {str(r["question_id"]): r for r in snap}
    order = [line.split(",", 1)[0] for line in
             (DATA / "BLIP" / "舊流程" / "exp2_success1000" / "dataset_manifest.csv").read_text(encoding="utf-8").splitlines()[1:]
             if line.strip()]
    missing = [q for q in order if q not in by_id]
    if missing:
        log.append(("無法重建 BLIP exp2 攻擊紀錄", f"{len(missing)} 題不在 2,298 題紀錄中", ""))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([by_id[q] for q in order], ensure_ascii=False, indent=2), encoding="utf-8")
    log.append(("重建（見說明）", "BLIP exp2 attack_dual_defense_log.json", target.relative_to(ws).as_posix()))


WS: Path | None = None  # set in main()


def copy(src: Path, dst: Path, log: list, how: str) -> None:
    _copy(src, dst)
    log.append((how, src.relative_to(REPO).as_posix(), dst.relative_to(WS).as_posix()))


README = """VLAttack 工作區（由 重算工具/build_workspace.py 產生，可隨時刪除重建）

這個資料夾把 repo 的程式與資料，放回原作者電腦上的排列（BLIP_attack/、BLIP_attack/output/、
BLIP_attack/output/VQA/），讓原始腳本不必修改路徑就能執行。repo 本身沒有被修改。
每個檔案放到哪裡，見同層的 workspace_manifest.tsv。

只要 CPU 就能跑的部分（用 repo 內已有的特徵與探針結果）：
  cd BLIP_attack/output/multi_captioner
  python analyze.py --exp exp1 exp2 exp3 --captioner blip --output-tag sigma08_fixed --fix-theta-star 0.8 --fix-theta-det 0.8
  （Florence-2：先設 VQA_STUDY=florence_vlattack，再加 --captioner florence2 --vqa-backend florence2）

需要 GPU、BLIP 權重與 COCO 圖片的部分（攻擊、圖片描述、模糊探針），另需設定：
  VLATTACK_IMG_ROOT       COCO 圖片資料夾（Florence 與題目難度腳本）
  VLATTACK_BLIP_PRETRAIN  BLIP model_base 權重（Florence 的 VLAttack 攻擊）
  VLATTACK_ANN_ROOT       VQAv2 標註資料夾（create_answer_list.py）
  VLATTACK_MINILM_DIR     sentence-transformers/all-MiniLM-L6-v2 的本機資料夾（run_rotation_cos_vs_ang.py）
  BLIP_attack/configs/VQA.yaml 的 vqa_root 改成自己的 COCO 位置
攻擊圖片（visual_results）不在 repo，需另外從 GitHub Release 下載解壓到對應的 visual_results/。

注意：
- BLIP exp2 的 attack_dual_defense_log.json 不在 repo，本工具依 exp2 的 dataset_manifest.csv 題號與順序，
  從 2,298 題攻擊紀錄重建。
- 實驗/程式/2_訓練/train_stage1_svm.py 是交付包自寫的腳本，直接在 repo 執行即可，不放進工作區。
"""


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    ws = Path(sys.argv[1]).resolve()
    if REPO in ws.parents or ws == REPO:
        sys.exit("工作區不可放在 repo 裡面，請指定 repo 外的資料夾")
    if ws.exists() and any(ws.iterdir()):
        sys.exit(f"{ws} 已存在且不是空的；請指定新的資料夾，或先刪除它")
    ws.mkdir(parents=True, exist_ok=True)
    global WS
    WS = ws
    log: list = []
    place_code(ws, log)
    place_data(ws, log)
    derive_blip_exp2_log(ws, log)
    with open(ws / "workspace_manifest.tsv", "w", encoding="utf-8", newline="") as f:
        f.write("方式\trepo 路徑\t工作區路徑\n")
        for row in log:
            f.write("\t".join(row) + "\n")
    (ws / "WORKSPACE_README.txt").write_text(README, encoding="utf-8")
    counts: dict[str, int] = {}
    for how, *_ in log:
        counts[how] = counts.get(how, 0) + 1
    print(f"工作區：{ws}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
