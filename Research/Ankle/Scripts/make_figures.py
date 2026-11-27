#!/usr/bin/env python3
"""
Ankle (Dataset304_AnkleCombined) 연구 페이지용 결과표(CSV) / 그림(PNG) 생성

  conda activate nnunetv2
  python make_figures.py                                   # 기본 경로 사용
  python make_figures.py --nnunet /home/ug/nnunet --out .. # out = Research/Ankle
  python make_figures.py --skip-3d                         # pyvista 3D 렌더 생략

입력 (서버 /home/ug/nnunet 기준)
  nnUNet_raw/Dataset304_AnkleCombined/{imagesTr,labelsTr,imagesTs,labelsTs}
  nnUNet_results/Dataset304_AnkleCombined/<trainer>/{fold_*/validation,crossval_results_folds_0_1_2_3_4}/summary.json
  predictions/Dataset304_Ts{,_fold0..4}/summary.json + *.nii.gz
  ankle_inference/{imagesTs,predictions_fold0}

출력
  Results/*.csv   (Dice 표, 임상 예측 부피)
  PNGs/*.png      (데이터셋 개요, 라벨별 Dice, Test GT vs Pred, 임상 2D/3D, 임상 오류 사례)

임상 케이스는 원본 파일명을 쓰지 않고 Clinical_01.. 로만 출력합니다 (정렬 순서 기준).

Dice 정의 (compare_folds.py의 fg(빈정답제외)와 동일)
  - 정답에 없는 라벨(n_ref == 0)은 해당 케이스-라벨 항목에서 제외
    (Utah는 tibia/talus/calcaneus만 라벨링, 나머지 뼈는 ignore(9))
  - 라벨별로 케이스 평균 -> 8개 라벨 평균 = "ignore-aware mean Dice"
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch
from scipy import ndimage as ndi
from skimage import measure

# ---------------------------------------------------------------- constants --
LABELS = {1: "tibia", 2: "fibula", 3: "talus", 4: "calcaneus",
          5: "tarsals", 6: "metatarsals", 7: "phalanges", 8: "femur"}
SHORT = {1: "Tib", 2: "Fib", 3: "Tal", 4: "Cal", 5: "Tar", 6: "Met", 7: "Pha", 8: "Fem"}
IGNORE = 9
# 검정 CT 배경 위 overlay용 (dark-surface steps). 해부학적으로 맞닿은 뼈끼리 구분되도록 배정.
COLORS = {1: "#338ced", 2: "#00a888", 3: "#d28e05", 4: "#eb6263",
          5: "#9887ef", 6: "#e24d8b", 7: "#0aac23", 8: "#e76624"}
IGNORE_COLOR = "#c2c2c2"
# 흰 배경 차트용
INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
SERIES = ["#257de0", "#f47230"]
HU_WINDOW = (-400, 1400)
TRAINER = "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "figure.facecolor": "white",
    "savefig.facecolor": "white", "hatch.color": IGNORE_COLOR, "hatch.linewidth": 0.8,
})


# ------------------------------------------------------------------ metrics --
def source_of(case):
    return "Utah" if case.startswith("UTA") else "VSD"


def load_summary(path):
    d = json.load(open(path))
    rows = []
    for c in d["metric_per_case"]:
        case = Path(c["reference_file"]).name.replace(".nii.gz", "")
        row = {"case": case, "source": source_of(case)}
        for k, name in LABELS.items():
            m = c["metrics"][str(k)]
            row[name] = m["Dice"] if m["n_ref"] > 0 else float("nan")
            row[f"_emptyfp_{name}"] = int(m["n_ref"] == 0 and m["n_pred"] > 0)
        vals = [row[n] for n in LABELS.values() if not math.isnan(row[n])]
        row["case_mean"] = float(np.mean(vals))
        rows.append(row)
    return rows, d["foreground_mean"]["Dice"]


def label_means(rows):
    out = {}
    for name in LABELS.values():
        v = [r[name] for r in rows if not math.isnan(r[name])]
        out[name] = float(np.mean(v)) if v else float("nan")
    out["mean_ignore_aware"] = float(np.nanmean([out[n] for n in LABELS.values()]))
    return out


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else
                            ("" if isinstance(v, float) else v)) for k, v in r.items()})


def build_tables(nn, out):
    res = nn / "nnUNet_results/Dataset304_AnkleCombined" / TRAINER
    pred = nn / "predictions"
    fields = ["run", "n_cases"] + list(LABELS.values()) + ["mean_ignore_aware", "mean_nnunet_default", "n_empty_fp"]

    def per_run(runs):
        table, cases = [], None
        for run, p in runs:
            rows, fg = load_summary(p)
            m = label_means(rows)
            m.update(run=run, n_cases=len(rows), mean_nnunet_default=fg,
                     n_empty_fp=sum(v for r in rows for k, v in r.items() if k.startswith("_emptyfp_")))
            table.append(m)
            cases = rows
        return table, cases

    cv_runs = [(f"fold_{i}", res / f"fold_{i}/validation/summary.json") for i in range(5)]
    cv_runs.append(("5-fold CV (all 61)", res / "crossval_results_folds_0_1_2_3_4/summary.json"))
    cv_table, cv_cases = per_run(cv_runs)

    ts_runs = [(f"fold_{i}", pred / f"Dataset304_Ts_fold{i}/summary.json") for i in range(5)]
    ts_runs.append(("5-fold ensemble", pred / "Dataset304_Ts/summary.json"))
    ts_table, ts_cases = per_run(ts_runs)

    write_csv(out / "Results/cv_per_fold.csv", cv_table, fields)
    write_csv(out / "Results/test_per_fold.csv", ts_table, fields)
    case_fields = ["case", "source"] + list(LABELS.values()) + ["case_mean"]
    write_csv(out / "Results/cv_per_case.csv", cv_cases, case_fields)
    write_csv(out / "Results/test_per_case.csv", ts_cases, case_fields)

    # source(Utah/VSD) 별 요약
    src_rows = []
    for split, rows in [("CV", cv_cases), ("Test", ts_cases)]:
        for src in ["Utah", "VSD"]:
            sub = [r for r in rows if r["source"] == src]
            m = label_means(sub)
            cm = [r["case_mean"] for r in sub]
            m.update(split=split, source=src, n_cases=len(sub),
                     case_mean=float(np.mean(cm)), case_sd=float(np.std(cm)), case_min=float(np.min(cm)))
            src_rows.append(m)
    write_csv(out / "Results/source_breakdown.csv", src_rows,
              ["split", "source", "n_cases"] + list(LABELS.values()) + ["mean_ignore_aware", "case_mean", "case_sd", "case_min"])

    # 라벨별 SD (CV / Test, n_ref>0 케이스)
    def sd(rows, name):
        v = [r[name] for r in rows if not math.isnan(r[name])]
        return float(np.std(v)), len(v)

    return cv_table[-1], ts_table[-1], {n: (sd(cv_cases, n), sd(ts_cases, n)) for n in LABELS.values()}, ts_cases


# ---------------------------------------------------------------- plot: dice --
def plot_per_label(cv, ts, sds, out):
    names = list(LABELS.values())
    rows = names + ["mean"]
    y = np.arange(len(rows))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0.94, 1.0)
    ax.set_ylim(-0.8, len(rows) - 0.4)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(y)
    ax.set_yticklabels([r if r != "mean" else "Mean (8 bones)" for r in rows], color=INK)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.axhline(0.5, color=AXIS, linewidth=0.8)

    off = 0.14
    for i, (key, d, lab) in enumerate([(0, cv, "5-fold CV  (n=61, train/val)"),
                                       (1, ts, "Held-out test  (n=7, 5-fold ensemble)")]):
        vals = [d[n] for n in names] + [d["mean_ignore_aware"]]
        yy = y + (off if i == 0 else -off)
        if i == 0:  # CV SD whisker (thin)
            for n, yv in zip(names, yy):
                s = sds[n][0][0]
                ax.plot([d[n] - s, min(d[n] + s, 1.0)], [yv, yv], color=SERIES[i], linewidth=1.2, alpha=0.45,
                        solid_capstyle="round")
        ax.scatter(vals, yy, s=46, color=SERIES[i], edgecolor="white", linewidth=1.6, zorder=3, label=lab)
    # selective direct labels: mean row + lowest bone
    for i, d in enumerate([cv, ts]):
        yv = y[-1] + (off if i == 0 else -off)
        ax.text(d["mean_ignore_aware"] + 0.0012, yv, f"{d['mean_ignore_aware']:.4f}", va="center",
                fontsize=8, color=INK)
    ax.set_xlabel("Dice (ignore-aware: labels absent from the ground truth are excluded)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8,
              handletextpad=0.3, columnspacing=1.4)
    fig.text(0.01, 0.005, "Whisker = CV ±1 SD.  Utah cases contribute to tibia / talus / calcaneus only.",
             fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "PNGs/Per_Label_Dice.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------ plot: slices --
def load(path, dtype=None):
    img = nib.load(str(path))
    assert nib.aff2axcodes(img.affine) == ("L", "P", "S"), f"{path}: expected LPS"
    a = np.asanyarray(img.dataobj)
    return (a.astype(dtype) if dtype else a), np.array(img.header.get_zooms()[:3], float)


def slice2d(vol, plane, idx):
    """LPS 배열 -> 화면용 2D (위=Superior). coronal: 왼쪽=환자 Right(radiological), sagittal: 왼쪽=Anterior"""
    s = vol[:, idx, :] if plane == "cor" else vol[idx, :, :]
    return s.T[::-1]


def largest_cc(mask):
    cc, n = ndi.label(mask)
    if n <= 1:
        return mask
    return cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)


def pick_slices(lab):
    """coronal: tibia+fibula 면적 최대 slice / sagittal: talus(최대 성분) 중심. 반대쪽 다리 성분은 무시."""
    shank = largest_cc(lab == 1) | (lab == 2) if (lab == 1).any() else lab > 0
    y_cor = int(np.argmax(shank.sum(axis=(0, 2))))
    tal = largest_cc(lab == 3) if (lab == 3).any() else lab > 0
    x_sag = int(round(ndi.center_of_mass(tal)[0]))
    return y_cor, x_sag


def crop_box(mask2d, margin_px, shape):
    if not mask2d.any():
        return 0, shape[0], 0, shape[1]
    r, c = np.where(mask2d)
    return (max(r.min() - margin_px[0], 0), min(r.max() + margin_px[0] + 1, shape[0]),
            max(c.min() - margin_px[1], 0), min(c.max() + margin_px[1] + 1, shape[1]))


def draw_overlay(ax, ct2d, lab2d, px, title, labels_txt=True, show_ignore=True):
    """px = (row_mm, col_mm). extent in mm so every panel shares one physical scale."""
    h, w = ct2d.shape
    ext = [0, w * px[1], h * px[0], 0]
    ax.imshow(np.clip(ct2d, *HU_WINDOW), cmap="gray", vmin=HU_WINDOW[0], vmax=HU_WINDOW[1],
              extent=ext, interpolation="bilinear")
    rgba = np.zeros((h, w, 4))
    for k, col in COLORS.items():
        m = lab2d == k
        if m.any():
            rgba[m] = list(matplotlib.colors.to_rgb(col)) + [0.42]
    ax.imshow(rgba, extent=ext, interpolation="nearest")
    yy, xx = (np.arange(h) + 0.5) * px[0], (np.arange(w) + 0.5) * px[1]
    for k, col in COLORS.items():
        m = lab2d == k
        if m.any():
            ax.contour(xx, yy, m.astype(float), levels=[0.5], colors=[col], linewidths=0.8)
    if show_ignore and (lab2d == IGNORE).any():
        m = (lab2d == IGNORE).astype(float)
        ax.contourf(xx, yy, m, levels=[0.5, 1.5], colors="none", hatches=["////"])
        ax.contour(xx, yy, m, levels=[0.5], colors=[IGNORE_COLOR], linewidths=0.6)
    if labels_txt:
        for k in LABELS:
            m = lab2d == k
            if m.sum() < 60:
                continue
            dt = ndi.distance_transform_edt(m)
            r, c = np.unravel_index(np.argmax(dt), dt.shape)
            ax.text((c + 0.5) * px[1], (r + 0.5) * px[0], SHORT[k], color="white", fontsize=6.5,
                    ha="center", va="center", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])
    ax.set_title(title, fontsize=8.5, color=INK, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return w * px[1], h * px[0]


def row_figure(panels, out_path, suptitle=None, legend_ignore=False, height_in=6.0, note=None):
    """panels: list of dict(ct, lab, px, title, [labels_txt]). 같은 mm 스케일로 한 줄 배치."""
    widths = [p["ct"].shape[1] * p["px"][1] for p in panels]
    heights = [p["ct"].shape[0] * p["px"][0] for p in panels]
    hmax = max(heights)
    fig_w = height_in * sum(widths) / hmax + 0.25 * len(panels)
    fig = plt.figure(figsize=(max(fig_w, 6.5), height_in + 1.1))
    gs = fig.add_gridspec(1, len(panels), width_ratios=widths, wspace=0.06,
                          left=0.01, right=0.99, top=0.9 if suptitle else 0.95, bottom=0.11)
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        draw_overlay(ax, p["ct"], p["lab"], p["px"], p["title"], p.get("labels_txt", True))
        ax.set_xlim(0, widths[i])
        ax.set_ylim(hmax * 1.07, 0)  # 위 정렬, 같은 스케일, 아래에 scale bar 자리
        ax.set_facecolor("white")
        scale_bar_at(ax, heights[i], hmax)
    handles = [Patch(facecolor=COLORS[k], edgecolor=COLORS[k], label=LABELS[k]) for k in LABELS]
    if legend_ignore:
        handles.append(Patch(facecolor="white", edgecolor=INK2, hatch="////", label="ignore (unlabeled bone)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7.5,
               handlelength=1.1, handletextpad=0.4, columnspacing=1.0, bbox_to_anchor=(0.5, 0.0))
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, color=INK, y=0.985)
    if note:
        fig.text(0.5, 0.055, note, ha="center", fontsize=7, color=INK2)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def scale_bar_at(ax, panel_h, hmax, length_mm=50):
    """패널 아래 흰 여백에 scale bar (라벨/뼈와 겹치지 않게)"""
    x0, x1 = ax.get_xlim()
    bx, by = x0 + 0.04 * (x1 - x0), panel_h + 0.018 * hmax
    ax.plot([bx, bx + length_mm], [by, by], color=INK, linewidth=2.0, solid_capstyle="butt", clip_on=False)
    ax.text(bx + length_mm + 4, by, f"{length_mm} mm", color=INK2, fontsize=6.5, ha="left", va="center")


def panels_for(ct, lab, zooms, tag, crop=True, sl=None, which=("cor", "sag"), sel_lab=None):
    sel = sel_lab if sel_lab is not None else lab
    y_cor, x_sag = sl or pick_slices(sel)
    out = []
    for plane in which:
        idx = y_cor if plane == "cor" else x_sag
        c2, l2 = slice2d(ct, plane, idx), slice2d(lab, plane, idx)
        px = (zooms[2], zooms[0] if plane == "cor" else zooms[1])
        if crop:
            ref = slice2d(sel, plane, idx) > 0
            m = (int(20 / px[0]), int(20 / px[1]))
            r0, r1, c0, c1 = crop_box(ref, m, c2.shape)
            c2, l2 = c2[r0:r1, c0:c1], l2[r0:r1, c0:c1]
        out.append(dict(ct=c2, lab=l2, px=px, title=f"{tag} | {'coronal' if plane == 'cor' else 'sagittal'}"))
    return out


# ------------------------------------------------------------------ figures --
def fig_dataset_overview(nn, out):
    raw = nn / "nnUNet_raw/Dataset304_AnkleCombined"
    panels = []
    for case, tag in [("UTA01L", "Utah (UTA01L)"), ("VSD002L", "VSD (VSD002L)")]:
        ct, z = load(raw / f"imagesTr/{case}_0000.nii.gz")
        lab, _ = load(raw / f"labelsTr/{case}.nii.gz", np.uint8)
        sel = np.where(lab == IGNORE, 0, lab)
        panels += panels_for(ct, lab, z, tag, crop=False, sel_lab=sel)
    row_figure(panels, out / "PNGs/Dataset_Overview.png", legend_ignore=True, height_in=6.4,
               suptitle="Ground truth by source - same physical scale (full FOV)")


def fig_test(nn, out, ts_cases):
    raw = nn / "nnUNet_raw/Dataset304_AnkleCombined"
    pred = nn / "predictions/Dataset304_Ts"
    by_case = {r["case"]: r for r in ts_cases}
    vsd = min((r for r in ts_cases if r["source"] == "VSD"), key=lambda r: r["case_mean"])["case"]
    for case, fname, extra in [(vsd, "Test_VSD_GT_vs_Pred.png", "lowest test case"),
                               ("UTA02L", "Test_Utah_GT_vs_Pred.png", "Dice on tibia/talus/calcaneus only")]:
        ct, z = load(raw / f"imagesTs/{case}_0000.nii.gz")
        gt, _ = load(raw / f"labelsTs/{case}.nii.gz", np.uint8)
        pr, _ = load(pred / f"{case}.nii.gz", np.uint8)
        sel = np.where(gt == IGNORE, 1, gt)  # ignore 포함 영역 기준 crop
        sl = pick_slices(np.where(gt == IGNORE, 0, gt))
        p_gt = panels_for(ct, gt, z, "GT", sl=sl, sel_lab=sel)
        p_pr = panels_for(ct, pr, z, "Prediction", sl=sl, sel_lab=sel)
        panels = [p_gt[0], p_pr[0], p_gt[1], p_pr[1]]
        row_figure(panels, out / f"PNGs/{fname}", legend_ignore=case.startswith("UTA"), height_in=6.0,
                   suptitle=f"Held-out test {case} ({extra}) - case mean Dice {by_case[case]['case_mean']:.4f}")


def clinical_cases(nn):
    d = nn / "ankle_inference"
    files = sorted((d / "predictions_fold0").glob("*.nii.gz"))
    return [(f"Clinical_{i + 1:02d}", d / "imagesTs" / f.name.replace(".nii.gz", "_0000.nii.gz"), f)
            for i, f in enumerate(files)]


def fig_clinical(nn, out):
    panels, vol_rows, errors = [], [], []
    for cid, img_p, pred_p in clinical_cases(nn):
        ct, z = load(img_p)
        pr, _ = load(pred_p, np.uint8)
        vv = float(np.prod(z)) / 1000.0
        row = {"case": cid, "spacing_mm": "x".join(f"{s:.3f}" for s in z),
               "fov_mm": "x".join(f"{s * n:.0f}" for s, n in zip(z, ct.shape))}
        for k, name in LABELS.items():
            cc, n = ndi.label(pr == k)
            sizes = np.sort(np.bincount(cc.ravel())[1:])[::-1] * vv if n else np.array([0.0])
            row[f"{name}_mL"] = float(sizes.sum())
            row[f"{name}_n_comp"] = int(n)
            row[f"{name}_extra_mL"] = float(sizes[1:].sum()) if n > 1 else 0.0
            if k in (1, 2, 3, 4, 8) and n > 1 and sizes[1] > 0.5:
                j = int(np.argsort(np.bincount(cc.ravel())[1:])[::-1][1]) + 1
                errors.append((sizes[1], cid, k, ndi.center_of_mass(cc == j)))
        vol_rows.append(row)
        # 대상 다리(가장 큰 tibia 성분 기준)로 slice 선택
        panels += panels_for(ct, pr, z, cid)
    fields = ["case", "spacing_mm", "fov_mm"] + [f"{n}_{s}" for n in LABELS.values() for s in ("mL", "n_comp", "extra_mL")]
    write_csv(out / "Results/clinical_volumes.csv", vol_rows, fields)
    row_figure(panels, out / "PNGs/Clinical_Inference_2D.png", height_in=6.4,
               suptitle="Private clinical CT - fold 0 model, no post-processing (no ground truth)")

    # 오류 사례: 케이스별 가장 큰 2차 성분 1개씩
    best = {}
    for size, cid, k, com in sorted(errors, reverse=True):
        best.setdefault(cid, (size, k, com))
    epanels = []
    lookup = {cid: (i, p) for cid, i, p in clinical_cases(nn)}
    for cid, (size, k, com) in sorted(best.items(), key=lambda t: -t[1][0])[:2]:
        ct, z = load(lookup[cid][0])
        pr, _ = load(lookup[cid][1], np.uint8)
        y = int(round(com[1]))
        c2, l2 = slice2d(ct, "cor", y), slice2d(pr, "cor", y)
        px = (z[2], z[0])
        r = c2.shape[0] - 1 - int(round(com[2]))
        c = int(round(com[0]))
        hr, hc = int(70 / px[0]), int(70 / px[1])
        r0, r1 = max(r - hr, 0), min(r + hr, c2.shape[0])
        c0, c1 = max(c - hc, 0), min(c + hc, c2.shape[1])
        epanels.append(dict(ct=c2[r0:r1, c0:c1], lab=l2[r0:r1, c0:c1], px=px,
                            title=f"{cid} | coronal\nextra '{LABELS[k]}' component ({size:.1f} mL)"))
    if epanels:
        row_figure(epanels, out / "PNGs/Clinical_Error_Cases.png", height_in=4.2,
                   suptitle="Largest secondary components (candidates for post-processing)")


def fig_clinical_3d(nn, out):
    import pyvista as pv
    pv.OFF_SCREEN = True
    cases = clinical_cases(nn)
    pl = pv.Plotter(off_screen=True, shape=(1, len(cases)), window_size=(600 * len(cases), 1100),
                    border=False)
    for i, (cid, _, pred_p) in enumerate(cases):
        pr, z = load(pred_p, np.uint8)
        pl.subplot(0, i)
        pl.set_background("#191919")
        for k, col in COLORS.items():
            m = pr == k
            if not m.any():
                continue
            sl = ndi.find_objects(m.astype(np.uint8))[0]
            sub = m[sl][::2, ::2, ::2].astype(np.float32)
            sub = np.pad(sub, 1)
            sub = ndi.gaussian_filter(sub, 0.8)
            origin = [(s.start - 2) * zz for s, zz in zip(sl, z)]
            grid = pv.ImageData(dimensions=sub.shape, spacing=tuple(z * 2), origin=origin)
            grid.point_data["m"] = sub.ravel(order="F")
            surf = grid.contour([0.5], scalars="m")
            if surf.n_points == 0:
                continue
            pl.add_mesh(surf.smooth_taubin(n_iter=30), color=col, smooth_shading=True,
                        specular=0.25, ambient=0.15)
        # 전면(anterior) 시점: LPS에서 anterior = -y
        pl.view_vector((0, -1, 0), viewup=(0, 0, 1))
        pl.camera.zoom(1.6)
    img = pl.screenshot(return_img=True)
    pl.close()
    bg = np.array([25, 25, 25])
    fgmask = np.abs(img[..., :3].astype(int) - bg).sum(-1) > 30
    fgmask[:40] = False  # 케이스 제목 텍스트 영역은 crop 계산에서 제외
    r = np.where(fgmask.any(1))[0]
    img = img[max(r.min() - 60, 0):min(r.max() + 30, img.shape[0])]
    fig = plt.figure(figsize=(img.shape[1] / 200, img.shape[0] / 200 + 0.45))
    ax = fig.add_axes([0, 0.45 / (img.shape[0] / 200 + 0.45), 1, 1 - 0.45 / (img.shape[0] / 200 + 0.45)])
    ax.imshow(img)
    ax.axis("off")
    for i, (cid, _, _) in enumerate(cases):
        ax.text((i + 0.03) * img.shape[1] / len(cases), 25, cid, color="#efefef", fontsize=10, va="top")
    handles = [Patch(facecolor=COLORS[k], edgecolor=COLORS[k], label=LABELS[k]) for k in LABELS]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=7.5,
               handlelength=1.1, handletextpad=0.4, columnspacing=1.0)
    fig.savefig(out / "PNGs/Clinical_Inference_3D.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnunet", default="/home/ug/nnunet", type=Path)
    ap.add_argument("--out", default=Path(__file__).resolve().parents[1], type=Path)
    ap.add_argument("--skip-3d", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="tables dataset test clinical clinical3d 중 일부만")
    a = ap.parse_args()
    (a.out / "PNGs").mkdir(parents=True, exist_ok=True)
    (a.out / "Results").mkdir(parents=True, exist_ok=True)
    todo = set(a.only or ["tables", "dataset", "test", "clinical", "clinical3d"])
    cv, ts, sds, ts_cases = build_tables(a.nnunet, a.out)
    if "tables" in todo:
        plot_per_label(cv, ts, sds, a.out)
        print(f"CV ignore-aware {cv['mean_ignore_aware']:.4f} (nnU-Net default {cv['mean_nnunet_default']:.4f})")
        print(f"Test ignore-aware {ts['mean_ignore_aware']:.4f} (nnU-Net default {ts['mean_nnunet_default']:.4f})")
    if "dataset" in todo:
        fig_dataset_overview(a.nnunet, a.out)
    if "test" in todo:
        fig_test(a.nnunet, a.out, ts_cases)
    if "clinical" in todo:
        fig_clinical(a.nnunet, a.out)
    if "clinical3d" in todo and not a.skip_3d:
        fig_clinical_3d(a.nnunet, a.out)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
