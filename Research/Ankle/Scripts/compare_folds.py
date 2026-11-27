#!/usr/bin/env python3
"""
nnU-Net v2 summary.json 여러 개를 fold별로 비교하는 스크립트

사용법:
  # 1) fold별 validation 결과 비교
  python compare_folds.py $RES/fold_*/validation/summary.json

  # 2) fold별 test 결과 비교 (fold마다 따로 predict + evaluate 한 뒤)
  python compare_folds.py /home/ug/nnunet/predictions/Dataset304_Ts_fold*/summary.json

출력 열:
  L1..Ln      : 클래스별 Dice 평균 (정답이 비어 있는 케이스는 제외)
  fg(nnUNet)  : nnU-Net이 계산한 foreground_mean Dice (정답이 비었는데 예측하면 0으로 포함)
  fg(빈정답제외): 정답이 비어 있는 케이스-클래스 항목을 뺀 전경 평균 Dice
  빈정답FP    : 정답에 없는 라벨을 예측한 케이스-클래스 개수
"""
import json
import math
import re
import sys
from pathlib import Path


def run_name(path: Path) -> str:
    m = re.search(r"fold_?([A-Za-z0-9]+)", str(path))
    return f"fold_{m.group(1)}" if m else path.parent.name


def label_key(k: str):
    try:
        return (0, int(k))
    except ValueError:
        return (1, k)


def nanmean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(path: Path):
    d = json.load(open(path))
    cases = d["metric_per_case"]
    labels = sorted(d["mean"].keys(), key=label_key)

    per_label = {}
    empty_fp = 0
    for lab in labels:
        vals = []
        for c in cases:
            m = c["metrics"].get(lab)
            if m is None:
                continue
            if m["n_ref"] == 0:
                if m["n_pred"] > 0:
                    empty_fp += 1
                continue
            vals.append(m["Dice"])
        per_label[lab] = nanmean(vals)

    return {
        "name": run_name(path),
        "n_cases": len(cases),
        "labels": labels,
        "per_label": per_label,
        "fg_nnunet": d["foreground_mean"]["Dice"],
        "fg_excl": nanmean(list(per_label.values())),
        "empty_fp": empty_fp,
    }


def main(paths):
    if not paths:
        print(__doc__)
        sys.exit(1)

    rows = [summarize(Path(p)) for p in paths]
    labels = rows[0]["labels"]

    header = ["run", "n"] + [f"L{l}" for l in labels] + ["fg(nnUNet)", "fg(빈정답제외)", "빈정답FP"]
    print("\t".join(header))
    for r in rows:
        line = [r["name"], str(r["n_cases"])]
        line += [f"{r['per_label'][l]:.4f}" for l in labels]
        line += [f"{r['fg_nnunet']:.4f}", f"{r['fg_excl']:.4f}", str(r["empty_fp"])]
        print("\t".join(line))

    if len(rows) > 1:
        best_nn = max(rows, key=lambda r: r["fg_nnunet"])
        best_ex = max(rows, key=lambda r: r["fg_excl"])
        print()
        print(f"최고 fg(nnUNet)     : {best_nn['name']} ({best_nn['fg_nnunet']:.4f})")
        print(f"최고 fg(빈정답제외) : {best_ex['name']} ({best_ex['fg_excl']:.4f})")


if __name__ == "__main__":
    main(sys.argv[1:])