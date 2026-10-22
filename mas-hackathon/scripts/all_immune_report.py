"""Generate a scientific summary table and figure for the all_immune
benchmark JSON produced by ``maehd.benchmark_real``.

Outputs (next to the JSON):

* ``all_immune_summary.md``  — markdown table with mean ± SEM across seeds,
  best value per column highlighted in bold.
* ``all_immune_recall.png``  — grouped bar chart, per-screen recall by
  method with SEM error bars; classic vs independent groups separated.

Run with the project's .venv:

    cd mas-hackathon
    source .venv/bin/activate && unset PYTHONPATH
    python scripts/all_immune_report.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
JSON_PATH = REPO / "results" / "all_immune" / "benchmark_all_immune_achilles.json"
OUT_DIR = JSON_PATH.parent

SCREENS = ["IFNγ", "IL-2", "NK"]  # display labels for the three columns
SCREEN_KEYS = ["ifng", "il2", "nk"]    # short keys (for table)

# Display labels + grouping for the methods that appear in the JSON.
METHOD_LABELS = {
    "classic_max_poh":      ("classic_max_poh",      "multi-target"),
    "classic_mean_poh":     ("classic_mean_poh",     "multi-target"),
    "classic_coverage_poh": ("classic_coverage_poh", "multi-target"),
    "ind_poh":              ("ind_poh",              "independent"),
    "ind_ucb":              ("ind_ucb",              "independent"),
    "ind_thompson":         ("ind_thompson",         "independent"),
    "ind_random":           ("ind_random",           "independent"),
}


def aggregate(d: dict) -> dict[str, dict]:
    """Return {method: {recall_mean(K,), recall_sem(K,), mean_recall_mean,
    mean_recall_sem, hit_ratio_mean, hit_ratio_sem, queries_mean}}.

    SEM = standard error of the mean across the five seeds.
    """
    seeds = sorted(d.keys())
    methods = list(d[seeds[0]].keys())
    n = len(seeds)
    out: dict[str, dict] = {}
    for m in methods:
        recall = np.array([d[s][m]["final_recall"] for s in seeds])  # (n, K)
        hit_ratio = np.array([d[s][m]["hit_ratio"] for s in seeds])
        queries = np.array([d[s][m]["total_queries"] for s in seeds])
        mean_recall = recall.mean(axis=1)
        out[m] = dict(
            recall_mean=recall.mean(axis=0),
            recall_sem=recall.std(axis=0, ddof=1) / math.sqrt(n),
            mean_recall_mean=mean_recall.mean(),
            mean_recall_sem=mean_recall.std(ddof=1) / math.sqrt(n),
            hit_ratio_mean=hit_ratio.mean(),
            hit_ratio_sem=hit_ratio.std(ddof=1) / math.sqrt(n),
            queries_mean=queries.mean(),
        )
    return out


# --------------------------------------------------------------------------- #
# Markdown table                                                              #
# --------------------------------------------------------------------------- #


def render_markdown_table(agg: dict[str, dict]) -> str:
    methods = list(METHOD_LABELS.keys())
    # Find best (highest) per numeric column to bold.
    bests = {}
    for i, k in enumerate(SCREEN_KEYS):
        bests[k] = max(methods, key=lambda m: agg[m]["recall_mean"][i])
    bests["mean"] = max(methods, key=lambda m: agg[m]["mean_recall_mean"])
    bests["hit"] = max(methods, key=lambda m: agg[m]["hit_ratio_mean"])

    def cell(m, key, idx=None, bold_method=None):
        if key in SCREEN_KEYS:
            val = agg[m]["recall_mean"][idx]
            sem = agg[m]["recall_sem"][idx]
        elif key == "mean":
            val, sem = agg[m]["mean_recall_mean"], agg[m]["mean_recall_sem"]
        else:
            val, sem = agg[m]["hit_ratio_mean"], agg[m]["hit_ratio_sem"]
        text = f"{val:.3f} ± {sem:.3f}"
        return f"**{text}**" if m == bold_method else text

    header = (
        "| Method | Group | IFNγ recall | IL-2 recall | NK recall | "
        "**Mean recall** | Hit ratio | Queries |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    rows = [header]
    for m in methods:
        label, group = METHOD_LABELS[m]
        rows.append(
            "| `{lbl}` | {grp} | {c0} | {c1} | {c2} | {cm} | {ch} | {q:.0f} |".format(
                lbl=label,
                grp=group,
                c0=cell(m, "ifng", 0, bests["ifng"]),
                c1=cell(m, "il2", 1, bests["il2"]),
                c2=cell(m, "nk", 2, bests["nk"]),
                cm=cell(m, "mean", bold_method=bests["mean"]),
                ch=cell(m, "hit", bold_method=bests["hit"]),
                q=agg[m]["queries_mean"],
            )
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #


def make_figure(agg: dict[str, dict], out_path: Path) -> None:
    methods = list(METHOD_LABELS.keys())
    # Sort within each group by mean recall (descending) for readability.
    classic = sorted(
        [m for m in methods if METHOD_LABELS[m][1] == "multi-target"],
        key=lambda m: -agg[m]["mean_recall_mean"],
    )
    indep = sorted(
        [m for m in methods if METHOD_LABELS[m][1] == "independent"],
        key=lambda m: -agg[m]["mean_recall_mean"],
    )
    order = classic + indep
    labels = [METHOD_LABELS[m][0] for m in order]

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
    })

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[2.4, 1.0])
    )

    # ----- Panel A: per-screen recall -----
    n_methods = len(order)
    x = np.arange(n_methods)
    width = 0.26
    palette = ["#2b6cb0", "#9b2c2c", "#276749"]  # IFNγ blue, IL-2 red, NK green

    for i, (label, color) in enumerate(zip(SCREENS, palette)):
        means = np.array([agg[m]["recall_mean"][i] for m in order])
        sems = np.array([agg[m]["recall_sem"][i] for m in order])
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=sems,
            capsize=3,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=label,
            error_kw=dict(lw=0.8),
        )

    # Vertical separator between multi-target and independent groups.
    sep = len(classic) - 0.5
    ax.axvline(sep, color="grey", lw=0.7, ls="--", alpha=0.7)
    ax.text(
        (len(classic) - 1) / 2,
        ax.get_ylim()[1] * 0.95,
        "multi-target GP\n(shared surrogate)",
        ha="center", va="top", fontsize=9, color="dimgray",
    )
    ax.text(
        sep + len(indep) / 2,
        ax.get_ylim()[1] * 0.95,
        "independent baselines\n(no target sharing)",
        ha="center", va="top", fontsize=9, color="dimgray",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Final recall per screen")
    ax.set_title(
        "A. Per-screen recall after 15 acquisition cycles "
        "(5 seeds, batch 30, n=17 355 genes, Achilles features)"
    )
    ax.legend(title="Screen", loc="upper right", frameon=False)
    ax.set_ylim(0, max(ax.get_ylim()[1], 0.13))
    ax.grid(axis="y", lw=0.4, alpha=0.4)

    # ----- Panel B: mean-recall summary -----
    mean_means = np.array([agg[m]["mean_recall_mean"] for m in order])
    mean_sems = np.array([agg[m]["mean_recall_sem"] for m in order])
    colors = ["#1a365d" if METHOD_LABELS[m][1] == "multi-target" else "#718096"
              for m in order]
    ax2.barh(
        np.arange(n_methods)[::-1],
        mean_means,
        xerr=mean_sems,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        capsize=3,
        error_kw=dict(lw=0.8),
    )
    ax2.set_yticks(np.arange(n_methods)[::-1])
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Mean recall across the three screens")
    ax2.set_title("B. Multi-target gain")
    ax2.grid(axis="x", lw=0.4, alpha=0.4)

    # Reference: ind_random
    rand_mean = agg["ind_random"]["mean_recall_mean"]
    ax2.axvline(rand_mean, color="black", lw=0.7, ls=":", label="random baseline")
    ax2.legend(loc="lower right", frameon=False)

    fig.suptitle(
        "GeneDisco all_immune benchmark — multi-target GP policies vs "
        "independent baselines",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def main() -> None:
    if not JSON_PATH.exists():
        raise FileNotFoundError(JSON_PATH)
    data = json.load(open(JSON_PATH))
    agg = aggregate(data)

    md = render_markdown_table(agg)
    md_path = OUT_DIR / "all_immune_summary.md"
    md_path.write_text(
        "# all_immune benchmark summary\n\n"
        "GeneDisco multi-target experiment: 3 immune screens "
        "(Schmidt 2021 IFNγ, Schmidt 2021 IL-2, Zhuang 2019 NK), "
        "n = 17 355 genes, Achilles features (PCA-20), batch 30, "
        "15 acquisition cycles, 5 seeds.\n\n"
        "Reported as mean ± SEM across seeds. **Bold** = best column.\n\n"
        + md + "\n"
    )
    fig_path = OUT_DIR / "all_immune_recall.png"
    make_figure(agg, fig_path)

    print("Wrote:")
    print(" ", md_path)
    print(" ", fig_path)


if __name__ == "__main__":
    main()
