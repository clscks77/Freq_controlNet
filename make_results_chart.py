"""
평가 결과를 3개의 개별 막대 그래프로 저장 (각각 1-column 크기).

사용법:
    python make_results_chart.py \
        --results_json "C:/Temp/TP2606_deepfake/evaluation/results.json" \
        --out_dir "C:/Temp/TP2606_deepfake/evaluation/charts"
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_bar(labels, values, title, ylabel, out_path,
             higher_better=False, fmt="{:.2f}", colors=None):
    if colors is None:
        colors = ["#7BA7BC", "#E07B54", "#6BAA75"]

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    bars = ax.bar(labels, values, color=colors,
                  edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(0, max(values) * 1.28)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=7)

    # 값 레이블
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                fmt.format(v), ha="center", va="bottom", fontsize=8)

    # 최고 성능 bar 강조
    best_idx = values.index(min(values)) if not higher_better else values.index(max(values))
    bars[best_idx].set_edgecolor("#222222")
    bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[✓] 저장: {out_path}")


def main(args):
    with open(args.results_json, encoding="utf-8") as f:
        data = json.load(f)

    keys   = list(data.keys())
    labels = ["Exp-0\n(Baseline)", "Exp-1\n(Type-A)", "Exp-2\n(Type-B)"]
    colors = ["#7BA7BC", "#E07B54", "#6BAA75"]

    fid     = [data[k]["FID"]            for k in keys]
    freq    = [data[k]["Freq_Dist_MAE"]  for k in keys]
    evasion = [data[k]["Evasion_Rate"] * 100 for k in keys]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_bar(labels, fid, "FID  ↓", "FID",
             out_dir / "chart_fid.png",
             higher_better=False, fmt="{:.1f}", colors=colors)

    save_bar(labels, freq, "Frequency Distance  ↓", "Ring [0,1,2] Amplitude MAE",
             out_dir / "chart_freq.png",
             higher_better=False, fmt="{:.1f}", colors=colors)

    save_bar(labels, evasion, "Evasion Rate  ↑", "Evasion Rate (%)",
             out_dir / "chart_evasion.png",
             higher_better=True, fmt="{:.1f}%", colors=colors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", required=True)
    parser.add_argument("--out_dir",      required=True)
    args = parser.parse_args()
    main(args)