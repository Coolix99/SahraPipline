from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy.stats import mannwhitneyu

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from config import scalar_path


DATA_DIR = scalar_path
WT_CSV = os.path.join(
    DATA_DIR,
    "membrane_dynamics_FINAL_200726_updated.csv"
)
SMOC_CSV = os.path.join(
    DATA_DIR,
    "Lucas_Vinita_smoc_merged_updated.csv"
)

FINAL_TIME = 144
ORDER = ["wt_dev", "wt_reg", "smoc_dev", "smoc_reg"]

LABELS = {
    "wt_dev": "WT dev",
    "smoc_dev": "Smoc dev",
    "wt_reg": "WT reg",
    "smoc_reg": "Smoc reg",
}

COLORS = {
    "wt_dev": "#2077b5",
    "smoc_dev": "#7b5ea7",
    "wt_reg": "#f57e1f",
    "smoc_reg": "#df3066",
}


mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "text.usetex": False,
    "axes.unicode_minus": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})


def set_plot_style():
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })


def normalize_condition(df):
    raw = (
        df.get(
            "condition",
            pd.Series(index=df.index, dtype="object")
        )
        .astype("string")
        .str.strip()
        .str.lower()
    )

    condition = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string"
    )

    condition[raw.str.contains("dev", na=False)] = "Development"
    condition[raw.str.contains("reg", na=False)] = "Regeneration"

    control = (
        df.get(
            "Is control",
            pd.Series(index=df.index, dtype="object")
        )
        .astype("string")
        .str.strip()
        .str.lower()
    )

    fallback = control.map({
        "true": "Development",
        "1": "Development",
        "yes": "Development",
        "false": "Regeneration",
        "0": "Regeneration",
        "no": "Regeneration",
    })

    return condition.fillna(fallback)


def load_final_areas(
    wt_csv=WT_CSV,
    smoc_csv=SMOC_CSV,
    final_time=FINAL_TIME
):
    wt = pd.read_csv(wt_csv)
    smoc = pd.read_csv(smoc_csv)

    frames = []

    for df, genotype, prefix in [
        (wt, "WT", "wt"),
        (smoc, "Smoc1_Smoc2", "smoc"),
    ]:
        df = df.copy()

        df["condition_clean"] = normalize_condition(df)
        df["time in hpf"] = pd.to_numeric(
            df["time in hpf"],
            errors="coerce"
        )
        df["Surface Area"] = pd.to_numeric(
            df["Surface Area"],
            errors="coerce"
        )

        df = df[
            df["genotype"]
            .astype("string")
            .str.strip()
            .str.casefold()
            == genotype.casefold()
        ]

        df = df[
            df["time in hpf"].eq(final_time)
            & df["condition_clean"].isin(
                ["Development", "Regeneration"]
            )
        ]

        df["group"] = (
            prefix
            + "_"
            + df["condition_clean"].map({
                "Development": "dev",
                "Regeneration": "reg",
            })
        )

        frames.append(df[["group", "Surface Area"]])

    data = pd.concat(frames, ignore_index=True).dropna()

    # Convert to units of (100 µm)^2.
    data["Surface Area"] /= 1e4

    return data


def summarize(data):
    return (
        data.groupby("group")["Surface Area"]
        .agg(
            mean="mean",
            sem="sem",
            std="std",
            n="size",
        )
        .reindex(ORDER)
    )


def clean_yaxis(ax):
    ymin, ymax = ax.get_ylim()
    rough = max((ymax - max(0, ymin)) / 5, 1e-9)
    power = 10 ** np.floor(np.log10(rough))

    step = next(
        m * power
        for m in (1, 2, 5, 10)
        if m * power >= rough
    )

    ax.set_ylim(0, 15)
    ax.yaxis.set_major_locator(MultipleLocator(step))


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    order = np.argsort(p)
    running = 0.0

    for rank, index in enumerate(order):
        running = max(
            running,
            min((len(p) - rank) * p[index], 1.0)
        )
        adjusted[index] = running

    return adjusted


def significance_stars(p):
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "ns"


def pairwise_statistics(data):
    comparisons = [
        (ORDER[i], ORDER[j])
        for i in range(len(ORDER))
        for j in range(i + 1, len(ORDER))
    ]

    raw = []

    for a, b in comparisons:
        values_a = data.loc[
            data["group"] == a,
            "Surface Area"
        ].dropna()

        values_b = data.loc[
            data["group"] == b,
            "Surface Area"
        ].dropna()

        p = mannwhitneyu(
            values_a,
            values_b,
            alternative="two-sided",
            method="auto",
        ).pvalue

        raw.append(p)

    adjusted = holm_adjust(raw)

    return [
        (a, b, p, p_holm)
        for (a, b), p, p_holm
        in zip(comparisons, raw, adjusted)
    ]


def report_statistics(data):
    results = pairwise_statistics(data)

    print("\nFinal surface area at 144 hpf")

    for group, row in (
        summarize(data)
        .dropna(subset=["mean"])
        .iterrows()
    ):
        print(
            f"{LABELS[group]:<10} "
            f"n={int(row['n']):>2}, "
            f"mean={row['mean']:.3f}, "
            f"SEM={row['sem']:.3f}, "
            f"SD={row['std']:.3f}"
        )

    for a, b, p, p_holm in results:
        print(
            f"{LABELS[a]} vs {LABELS[b]}: "
            f"Mann-Whitney p={p:.4g}, "
            f"Holm p={p_holm:.4g} "
            f"({significance_stars(p_holm)})"
        )


def add_significance_brackets(
    ax,
    x,
    summary,
    results,
    error_column="sem"
):
    significant = [
        (a, b, p_holm)
        for a, b, _, p_holm in results
        if p_holm <= 0.05
    ]

    if not significant:
        return

    group_index = {
        group: i
        for i, group in enumerate(ORDER)
    }

    plot_tops = (
        summary["mean"] + summary[error_column]
    ).to_numpy(dtype=float)

    base_height = np.nanmax(plot_tops)
    y_range = max(base_height, 1e-9)

    step = 0.10 * y_range
    bracket_height = 0.025 * y_range

    significant.sort(
        key=lambda result: (
            group_index[result[1]]
            - group_index[result[0]],
            group_index[result[0]],
        )
    )

    for level, (a, b, p_holm) in enumerate(
        significant,
        start=1
    ):
        i = group_index[a]
        j = group_index[b]
        y = base_height + level * step

        ax.plot(
            [x[i], x[i], x[j], x[j]],
            [
                y,
                y + bracket_height,
                y + bracket_height,
                y,
            ],
            color="k",
            linewidth=1.0,
            clip_on=False,
        )

        ax.text(
            (x[i] + x[j]) / 2,
            y + bracket_height,
            significance_stars(p_holm),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylim(
        top=base_height
        + (len(significant) + 1.5) * step
    )


def format_group_axis(ax, x, final_time):
    ax.set_ylabel(r"A [$\,(100\,\mu m)^2$]")
    # ax.set_title(f"Surface Area at {final_time} hpf")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABELS[group] for group in ORDER],
        rotation=45,
        ha="right",
    )

    ax.grid(False)
    sns.despine(ax=ax)


def plot_final_area(data, final_time=FINAL_TIME):
    """
    Original bar chart: mean ± SEM.

    This function is unchanged in its visual output.
    """
    summary = summarize(data)
    results = pairwise_statistics(data)

    x = np.arange(len(ORDER)) * 1.7
    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    ax.bar(
        x,
        summary["mean"],
        color=[COLORS[group] for group in ORDER],
        width=0.75,
        edgecolor="none",
        zorder=2,
    )

    ax.errorbar(
        x,
        summary["mean"],
        yerr=summary["sem"],
        fmt="none",
        ecolor="k",
        elinewidth=1.2,
        capsize=3,
        zorder=4,
    )

    format_group_axis(ax, x, final_time)

    clean_yaxis(ax)

    add_significance_brackets(
        ax,
        x,
        summary,
        results,
        error_column="sem",
    )

    fig.tight_layout()

    return fig, ax


def plot_final_area_dot_whisker(
    data,
    final_time=FINAL_TIME,
    jitter_width=0.24,
    point_alpha=0.45,
    random_seed=42,
    reference_ax=None,
):
    """
    Show individual measurements, mean ± SD, and significance brackets.

    When reference_ax is provided, its y-limits and major tick spacing are
    applied so this figure matches the bar plot exactly.
    """
    summary = summarize(data)
    results = pairwise_statistics(data)

    x = np.arange(len(ORDER)) * 1.7
    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    rng = np.random.default_rng(random_seed)

    for group_index, group in enumerate(ORDER):
        values = (
            data.loc[
                data["group"] == group,
                "Surface Area",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if values.size == 0:
            continue

        jitter = rng.uniform(
            -jitter_width,
            jitter_width,
            size=values.size,
        )

        ax.scatter(
            x[group_index] + jitter,
            values,
            s=25,
            color=COLORS[group],
            alpha=point_alpha,
            edgecolors="none",
            zorder=2,
        )

        ax.errorbar(
            x[group_index],
            summary.loc[group, "mean"],
            yerr=summary.loc[group, "std"],
            fmt="o",
            markersize=7,
            color=COLORS[group],
            ecolor=COLORS[group],
            markerfacecolor=COLORS[group],
            markeredgecolor=COLORS[group],
            elinewidth=1.8,
            capsize=5,
            capthick=1.8,
            zorder=4,
        )

    format_group_axis(ax, x, final_time)

    add_significance_brackets(
        ax,
        x,
        summary,
        results,
        error_column="std",
    )

    ax.set_ylim(0, 15)
    ax.yaxis.set_major_locator(MultipleLocator(5))

    fig.tight_layout()

    return fig, ax

def save_figure(fig, output_base, dpi=300):
    output_base = Path(output_base)
    output_base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for suffix in ("svg", "pdf"):
        fig.savefig(
            output_base.with_suffix(f".{suffix}"),
            bbox_inches="tight",
        )

    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
    )


def main(wt_csv=WT_CSV, smoc_csv=SMOC_CSV):
    set_plot_style()

    data = load_final_areas(
        wt_csv,
        smoc_csv,
    )

    report_statistics(data)

    # Original bar figure: mean ± SEM.
    bar_fig, bar_ax = plot_final_area(data)

    save_figure(
        bar_fig,
        Path("plots_WT_Smoc12")
        / "final_surface_area",
    )

    # New dot-and-whisker figure, using the same y-axis as the bar plot.
    dot_fig, _ = plot_final_area_dot_whisker(
        data,
        reference_ax=bar_ax,
    )

    save_figure(
        dot_fig,
        Path("plots_WT_Smoc12")
        / "final_surface_area_dot_whisker",
    )

    plt.show()


if __name__ == "__main__":
    main()