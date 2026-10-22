"""Generate a scientific summary table and figure for the diverse-setup
benchmark JSON produced by ``ai.experiments.compare_real``.

The diverse setup is the K=4 maximally-heterogeneous multi-target case:
IFNγ + IL-2 + Tau + SARS-CoV-2.  Four methods are compared:

    random       — uniform random selection
    parego       — Chebyshev-scalarized GP (paper baseline)
    singleagent  — one LLM agent seeing all targets   (ablation)
    multiagent   — K specialist LLMs + moderator       (main method)

Outputs next to the JSON:

* ``diverse_summary.md``  — markdown table, mean ± SEM, best per column bold.
* ``diverse_recall.png``  — two-panel figure: per-screen grouped bars + mean.

Run with the project's .venv:

    cd mas-hackathon
    unset PYTHONPATH && .venv/bin/python scripts/diverse_report.py \
        --json results/diverse_compare/diverse_gp.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Display labels for the four diverse-setup screens, in the canonical
# (IFNG, IL2, TAU, SARS) order used by MULTI_TARGET_SETUPS["diverse"].
SCREEN_DISPLAY = {
    "schmidt_2021_ifng":  "IFNγ",
    "schmidt_2021_il2":   "IL-2",
    "sanchez_2021_tau":   "Tau",
    "zhu_2021_sarscov2":  "SARS-CoV-2",
}

METHOD_ORDER = ["random", "parego", "singleagent", "multiagent"]
METHOD_LABELS = {
    "random":       ("random",      "baseline"),
    "parego":       ("ParEGO",      "GP"),
    "singleagent":  ("SingleAgent", "LLM"),
    "multiagent":   ("MultiAgent",  "LLM (multi)"),
}
GROUP_COLOR = {
    "baseline":     "#a0aec0",  # neutral grey
    "GP":           "#2b6cb0",  # blue
    "LLM":          "#9b2c2c",  # red
    "LLM (multi)":  "#276749",  # green
}


def load_results(json_path: Path) -> tuple[list[dict], dict]:
    blob = json.load(open(json_path))
    return blob["results"], blob.get("config", {})


def aggregate(results: list[dict]) -> tuple[list[str], dict[str, dict]]:
    """Group runs by method.  Returns (screen_names, agg).

    agg[method] = dict(
        recall_mean(K,), recall_sem(K,),
        mean_recall_mean, mean_recall_sem,
        n_seeds, elapsed_mean,
    )

    SEM = std(ddof=1) / sqrt(n).  When only one seed is present the SEM is
    reported as 0 (and flagged in the caption).
    """
    by_method: dict[str, list[dict]] = defaultdict(list)
    screen_names: list[str] = []
    for r in results:
        if not r.get("success", True):
            continue
        by_method[r["method"]].append(r)
        if not screen_names and r.get("screen_names"):
            screen_names = r["screen_names"]

    agg: dict[str, dict] = {}
    K = len(screen_names)
    for method, runs in by_method.items():
        per_target = np.array(
            [r["final_per_target_recall"] for r in runs]
        )  # (n_seeds, K)
        mean_recall = np.array([r["final_mean_recall"] for r in runs])
        elapsed = np.array([r.get("elapsed_sec", float("nan")) for r in runs])
        n = len(runs)
        if n > 1:
            recall_sem = per_target.std(axis=0, ddof=1) / math.sqrt(n)
            mean_sem = float(mean_recall.std(ddof=1) / math.sqrt(n))
        else:
            recall_sem = np.zeros(K)
            mean_sem = 0.0
        agg[method] = dict(
            recall_mean=per_target.mean(axis=0),
            recall_sem=recall_sem,
            mean_recall_mean=float(mean_recall.mean()),
            mean_recall_sem=mean_sem,
            n_seeds=n,
            elapsed_mean=float(np.nanmean(elapsed)),
        )
    return screen_names, agg


# --------------------------------------------------------------------------- #
# Markdown table                                                              #
# --------------------------------------------------------------------------- #


def render_markdown_table(
    agg: dict[str, dict], screen_names: list[str], config: dict
) -> str:
    methods = [m for m in METHOD_ORDER if m in agg]
    K = len(screen_names)
    cols = [SCREEN_DISPLAY.get(s, s) for s in screen_names]

    # best-per-column highlighting
    bests = {}
    for i in range(K):
        bests[i] = max(methods, key=lambda m: agg[m]["recall_mean"][i])
    bests["mean"] = max(methods, key=lambda m: agg[m]["mean_recall_mean"])

    def fmt(val, sem, bold):
        text = f"{val:.3f} ± {sem:.3f}"
        return f"**{text}**" if bold else text

    header = (
        "| Method | Group | "
        + " | ".join(f"{c} recall" for c in cols)
        + " | **Mean recall** | n seeds | Time (s) |\n"
        "|---|---|"
        + "|".join(["---"] * (K + 3))
        + "|"
    )
    rows = [header]
    for m in methods:
        label, group = METHOD_LABELS[m]
        per_target_cells = [
            fmt(
                agg[m]["recall_mean"][i],
                agg[m]["recall_sem"][i],
                bold=(m == bests[i]),
            )
            for i in range(K)
        ]
        mean_cell = fmt(
            agg[m]["mean_recall_mean"],
            agg[m]["mean_recall_sem"],
            bold=(m == bests["mean"]),
        )
        rows.append(
            "| `{lbl}` | {grp} | {pt} | {mc} | {ns} | {t:.0f} |".format(
                lbl=label,
                grp=group,
                pt=" | ".join(per_target_cells),
                mc=mean_cell,
                ns=agg[m]["n_seeds"],
                t=agg[m]["elapsed_mean"],
            )
        )

    # Note about missing methods.
    missing = [m for m in METHOD_ORDER if m not in agg]
    note_lines = []
    if missing:
        note_lines.append(
            f"_Methods not run in this evaluation: "
            f"{', '.join(METHOD_LABELS[m][0] for m in missing)} "
            f"(typically because OPENAI_API_KEY was not set when launched)._"
        )
    if note_lines:
        rows.append("")
        rows.extend(note_lines)
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #


def make_figure(
    agg: dict[str, dict],
    screen_names: list[str],
    config: dict,
    out_path: Path,
) -> None:
    methods = [m for m in METHOD_ORDER if m in agg]
    K = len(screen_names)
    labels = [METHOD_LABELS[m][0] for m in methods]
    screen_labels = [SCREEN_DISPLAY.get(s, s) for s in screen_names]

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
        1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[2.4, 1.0])
    )

    # ----- Panel A: per-screen recall, grouped bars (one group per screen) ----
    n_methods = len(methods)
    n_screens = K
    x = np.arange(n_screens)
    width = 0.8 / max(n_methods, 1)
    palette_a = [GROUP_COLOR[METHOD_LABELS[m][1]] for m in methods]

    for j, m in enumerate(methods):
        offsets = (j - (n_methods - 1) / 2) * width
        means = agg[m]["recall_mean"]
        sems = agg[m]["recall_sem"]
        ax.bar(
            x + offsets, means, width,
            yerr=sems, capsize=3,
            color=palette_a[j], edgecolor="black", linewidth=0.4,
            label=labels[j],
            error_kw=dict(lw=0.8),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(screen_labels)
    ax.set_ylabel("Final recall per screen")
    n_cyc = config.get("n_cycles", "?")
    bs = config.get("batch_size", "?")
    seeds = config.get("seeds", [])
    ax.set_title(
        f"A. Per-screen recall after {n_cyc} cycles "
        f"(batch={bs}, {len(seeds)} seed{'s' if len(seeds)!=1 else ''}, "
        f"diverse setup, K={K})"
    )
    ax.legend(title="Method", loc="upper right", frameon=False)
    ymax = max(0.05,
               max((agg[m]["recall_mean"].max() + agg[m]["recall_sem"].max())
                   for m in methods) * 1.25)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", lw=0.4, alpha=0.4)

    # ----- Panel B: mean recall across screens ---------------------------
    mean_means = np.array([agg[m]["mean_recall_mean"] for m in methods])
    mean_sems = np.array([agg[m]["mean_recall_sem"] for m in methods])
    colors = palette_a
    ypos = np.arange(n_methods)[::-1]
    ax2.barh(
        ypos, mean_means, xerr=mean_sems,
        color=colors, edgecolor="black", linewidth=0.4,
        capsize=3, error_kw=dict(lw=0.8),
    )
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Mean recall across all 4 screens")
    ax2.set_title("B. Average recall")
    ax2.grid(axis="x", lw=0.4, alpha=0.4)
    if "random" in agg:
        ax2.axvline(
            agg["random"]["mean_recall_mean"],
            color="black", lw=0.7, ls=":",
            label="random baseline",
        )
        ax2.legend(loc="lower right", frameon=False)

    fig.suptitle(
        "GeneDisco diverse benchmark — four-method comparison on K=4 "
        "heterogeneous screens (IFNγ + IL-2 + Tau + SARS-CoV-2)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results" / "diverse_compare" / "diverse_gp.json",
        help="Path to compare_real_results JSON.",
    )
    args = parser.parse_args()
    json_path: Path = args.json.resolve()
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    out_dir = json_path.parent
    results, config = load_results(json_path)
    screen_names, agg = aggregate(results)
    if not agg:
        raise RuntimeError(f"No successful runs found in {json_path}")

    md = render_markdown_table(agg, screen_names, config)
    md_path = out_dir / "diverse_summary.md"
    md_path.write_text(
        "# diverse benchmark summary (compare_real, four methods)\n\n"
        "GeneDisco multi-target experiment: 4 heterogeneous screens — "
        "Schmidt 2021 IFNγ, Schmidt 2021 IL-2, Sanchez 2021 Tau, "
        "Zhu 2021 SARS-CoV-2. PCA(20) features for GP methods; "
        f"BiologyEnv (no surrogate, pool sample) for LLM methods. "
        f"batch={config.get('batch_size','?')}, "
        f"n_cycles={config.get('n_cycles','?')}, "
        f"seeds={config.get('seeds','?')}.\n\n"
        "Reported as mean ± SEM across seeds. **Bold** = best column.\n\n"
        + md + "\n"
    )

    fig_path = out_dir / "diverse_recall.png"
    make_figure(agg, screen_names, config, fig_path)

    print("Wrote:")
    print(" ", md_path)
    print(" ", fig_path)


if __name__ == "__main__":
    main()
