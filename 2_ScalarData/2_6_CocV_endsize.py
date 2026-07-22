import os
import sys

from itertools import combinations
from scipy.stats import ttest_ind
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import scalar_path

mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "text.usetex": False,
    "axes.unicode_minus": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})

COND_ORDER = ["Development", "Regeneration", "4850cut", "7230cut"]
COND_LABEL = {
    "Development": "Development",
    "Regeneration": "Regeneration 30%",
    "4850cut": "Regeneration 50%",
    "7230cut": "Late Amputation 30%",
}
COND_COLOR = {
    "Development": "#2077b5",
    "Regeneration": "#f57e1f",
    "4850cut": "#592d10",
    "7230cut": "#fed1a4",
}


def set_plot_style_big():
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })


def clean_limits(vmin, vmax, step):
    return step * np.floor(vmin / step), step * np.ceil(vmax / step)


def set_clean_yaxis(ax, ymin=None, ymax=None, step=None):
    current_ymin, current_ymax = ax.get_ylim()
    ymin = current_ymin if ymin is None else ymin
    ymax = current_ymax if ymax is None else ymax

    if step is None:
        span = ymax - ymin
        if span <= 0:
            step = 1.0
        else:
            rough = span / 5.0
            power = 10 ** np.floor(np.log10(rough))
            for multiplier in (1, 2, 5, 10):
                if multiplier * power >= rough:
                    step = multiplier * power
                    break

    ymin, ymax = clean_limits(ymin, ymax, step)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(step))


def save_figure(fig, filename_base, dpi=300):
    fig.savefig(f"{filename_base}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{filename_base}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{filename_base}.png", format="png", dpi=dpi, bbox_inches="tight")


def cov(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    mean = np.mean(x)
    if mean == 0:
        return np.nan
    return np.std(x, ddof=1) / mean


def bootstrap_cov_sem(x, n_boot=5000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return cov(x), np.nan

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    boot_cov = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(indices, size=n, replace=True)
        boot_cov[i] = cov(x[sample])
    return cov(x), np.nanstd(boot_cov, ddof=1)


def summarize_cov_by_groups(df, value_col, group_cols, n_boot=5000, seed=0):
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = group[value_col].to_numpy(dtype=float)
        cov_value, cov_sem = bootstrap_cov_sem(values, n_boot=n_boot, seed=seed)
        rows.append(key_tuple + (cov_value, cov_sem, len(group)))
    return pd.DataFrame(rows, columns=[*group_cols, "cov", "cov_sem", "n"])


def plot_cov_grouped(
    df,
    value_col="Surface Area",
    time_col="time in hpf",
    condition_col="condition",
    default_times=(48, 144),
    special_times=None,
    conditions=COND_ORDER,
    n_boot=5000,
    seed=0,
    fig_size=None,
    style="bar",
    gap=0.8,
    within=0.9,
):
    special_times = {"7230cut": (72, 144)} if special_times is None else special_times
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if fig_size is None:
        fig_size = (5.6, 3.2) if style == "point" else (5.0, 3.6)

    fig, ax = plt.subplots(figsize=fig_size)
    x_pos, heights, yerr, colors = [], [], [], []
    condition_centers, tick_labels = [], []
    x_start = 0.0

    for condition in conditions:
        times = special_times.get(condition, default_times)
        subset = df[
            (df[condition_col] == condition) & (df[time_col].isin(times))
        ]
        if subset.empty:
            x_start += gap
            continue

        summary = summarize_cov_by_groups(
            subset,
            value_col=value_col,
            group_cols=[time_col],
            n_boot=n_boot,
            seed=seed,
        )
        condition_positions = []
        for i, time in enumerate(times):
            row = summary[summary[time_col] == time]
            if row.empty:
                continue
            position = x_start + i * within
            condition_positions.append(position)
            x_pos.append(position)
            heights.append(float(row["cov"].iloc[0]))
            yerr.append(float(row["cov_sem"].iloc[0]))
            colors.append(COND_COLOR[condition])

        if condition_positions:
            condition_centers.append(np.mean(condition_positions))
            tick_labels.append(COND_LABEL[condition])
            x_start = condition_positions[-1] + gap
        else:
            x_start += gap

    x_pos = np.asarray(x_pos)
    heights = np.asarray(heights)
    yerr = np.asarray(yerr)

    if style == "bar":
        ax.bar(
            x_pos,
            heights,
            color=colors,
            width=0.75,
            edgecolor="none",
            zorder=2,
        )
    else:
        ax.scatter(x_pos, heights, color=colors, s=35, zorder=3)

    ax.errorbar(
        x_pos,
        heights,
        yerr=yerr,
        fmt="none",
        ecolor="k",
        elinewidth=1.2,
        capsize=3,
        zorder=4,
    )
    symbol = "A" if value_col == "Surface Area" else "V"
    ax.set_ylabel(f"CoV ({symbol})")
    ax.set_title(f"{value_col} variability")
    ax.set_xticks(condition_centers)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    set_clean_yaxis(ax, ymin=0)
    sns.despine()
    ax.grid(False)
    fig.tight_layout()
    return fig, ax


def summarize_final_size(df, value_col, condition_col, conditions):
    rows = []
    for condition in conditions:
        values = df.loc[df[condition_col] == condition, value_col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
        rows.append((condition, mean, sem, len(values)))
    return pd.DataFrame(rows, columns=["condition", "mean", "sem", "n"])


def plot_final_size(
    df,
    value_col="Surface Area",
    time_col="time in hpf",
    condition_col="condition",
    final_time=144,
    conditions=COND_ORDER,
    fig_size=(5.0, 3.6),
    condition_spacing=1.7,
):
    subset = df[
        (df[time_col] == final_time) & (df[condition_col].isin(conditions))
    ].copy()
    subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")

    scale = 1e4 if value_col == "Surface Area" else 1.0
    subset[value_col] = subset[value_col] / scale
    summary = summarize_final_size(subset, value_col, condition_col, conditions)

    x_pos = np.arange(len(summary)) * condition_spacing
    heights = summary["mean"].to_numpy()
    yerr = summary["sem"].to_numpy()
    colors = [COND_COLOR[condition] for condition in summary["condition"]]
    labels = [COND_LABEL[condition] for condition in summary["condition"]]

    fig, ax = plt.subplots(figsize=fig_size)
    ax.bar(
        x_pos,
        heights,
        color=colors,
        width=0.75,
        edgecolor="none",
        zorder=2,
    )
    ax.errorbar(
        x_pos,
        heights,
        yerr=yerr,
        fmt="none",
        ecolor="k",
        elinewidth=1.2,
        capsize=3,
        zorder=4,
    )

    if value_col == "Surface Area":
        ax.set_ylabel(r"A [$\,(100\,\mu m)^2$]")
    else:
        ax.set_ylabel(r"V [$\mu m^3$]")
    ax.set_title(f"{value_col} at {final_time} hpf")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    set_clean_yaxis(ax, ymin=0)
    sns.despine()
    ax.grid(False)
    fig.tight_layout()
    return fig, ax


def load_growth_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "time in hpf" in df.columns:
        df["time in hpf"] = pd.to_numeric(df["time in hpf"], errors="coerce")
    if {"Volume", "Surface Area"}.issubset(df.columns):
        df["Mean Thickness"] = df["Volume"] / df["Surface Area"]
    return df

def significance_stars(p_value):
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p_values[valid])]
    running_max = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        corrected = min((m - rank) * p_values[index], 1.0)
        running_max = max(running_max, corrected)
        adjusted[index] = running_max
    return adjusted


def permutation_test_cov_decrease(early, final, n_perm=10000, seed=0):
    early = np.asarray(early, dtype=float)
    final = np.asarray(final, dtype=float)
    early = early[np.isfinite(early)]
    final = final[np.isfinite(final)]

    if len(early) < 2 or len(final) < 2:
        return np.nan, np.nan

    observed = cov(early) - cov(final)
    pooled = np.concatenate([early, final])
    n_early = len(early)
    rng = np.random.default_rng(seed)
    null_differences = []

    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        difference = cov(shuffled[:n_early]) - cov(shuffled[n_early:])
        if np.isfinite(difference):
            null_differences.append(difference)

    if not null_differences:
        return observed, np.nan

    null_differences = np.asarray(null_differences)
    p_value = (
        np.sum(null_differences >= observed) + 1
    ) / (len(null_differences) + 1)
    return observed, p_value


def report_bar_statistics(
    df,
    value_cols=("Volume", "Surface Area"),
    time_col="time in hpf",
    condition_col="condition",
    default_times=(48, 144),
    special_times=None,
    final_time=144,
    conditions=COND_ORDER,
    n_perm=10000,
    seed=0,
):
    special_times = {"7230cut": (72, 144)} if special_times is None else special_times

    print("\n" + "=" * 78)
    print("BAR SAMPLE SIZES AND STATISTICAL TESTS")
    print("=" * 78)

    for value_col in value_cols:
        data = df.copy()
        data[value_col] = pd.to_numeric(data[value_col], errors="coerce")

        print(f"\n{value_col.upper()}")
        print("-" * 78)
        print("CoV bars and one-sided tests for a decrease")

        for condition in conditions:
            early_time, late_time = special_times.get(condition, default_times)

            early = data.loc[
                (data[condition_col] == condition)
                & (data[time_col] == early_time),
                value_col,
            ].dropna().to_numpy()

            late = data.loc[
                (data[condition_col] == condition)
                & (data[time_col] == late_time),
                value_col,
            ].dropna().to_numpy()

            if len(early) == 0 and len(late) == 0:
                continue

            difference, p_value = permutation_test_cov_decrease(
                early,
                late,
                n_perm=n_perm,
                seed=seed,
            )

            early_cov = cov(early)
            late_cov = cov(late)
            label = COND_LABEL.get(condition, condition)

            print(
                f"  {label:<24} "
                f"{early_time:>3} hpf: n={len(early):<3} CoV={early_cov:.4f} | "
                f"{late_time:>3} hpf: n={len(late):<3} CoV={late_cov:.4f} | "
                f"decrease={difference:.4f}, "
                f"p={p_value:.4g} ({significance_stars(p_value)})"
            )

        print(f"\nFinal-size bars at {final_time} hpf")
        final_groups = {}

        for condition in conditions:
            values = data.loc[
                (data[condition_col] == condition)
                & (data[time_col] == final_time),
                value_col,
            ].dropna().to_numpy()

            if len(values) == 0:
                continue

            final_groups[condition] = values
            label = COND_LABEL.get(condition, condition)
            print(f"  {label:<24} n={len(values)}")

        comparisons = []
        for first, second in combinations(final_groups, 2):
            result = ttest_ind(
                final_groups[first],
                final_groups[second],
                equal_var=False,
                nan_policy="omit",
            )
            comparisons.append((first, second, float(result.pvalue)))

        if comparisons:
            adjusted = holm_adjust([item[2] for item in comparisons])

            print("\nPairwise final-size tests: Welch t-test, Holm-adjusted")
            for (first, second, raw_p), adjusted_p in zip(
                comparisons,
                adjusted,
            ):
                first_label = COND_LABEL.get(first, first)
                second_label = COND_LABEL.get(second, second)

                print(
                    f"  {first_label:<24} vs {second_label:<24} "
                    f"p={raw_p:.4g}, "
                    f"p_holm={adjusted_p:.4g} "
                    f"({significance_stars(adjusted_p)})"
                )
        else:
            print("  Not enough groups for pairwise testing.")

    print(
        "\nSignificance: ns >= 0.05, * < 0.05, ** < 0.01, "
        "*** < 0.001, **** < 0.0001"
    )
    print("=" * 78)

def main():
    set_plot_style_big()
    csv_file = os.path.join(scalar_path, "membrane_dynamics_FINAL_200726_updated.csv")
    df = load_growth_csv(csv_file)
    report_bar_statistics(df)
    print("\nAvailable times:")
    print(sorted(df["time in hpf"].dropna().unique()))
    print("\nAvailable conditions:")
    print(sorted(df["condition"].dropna().unique()))

    out_dir = os.path.join(".", "plots_cov_mean")
    os.makedirs(out_dir, exist_ok=True)

    for value_col in ("Volume", "Surface Area"):
        cov_fig, _ = plot_cov_grouped(
            df,
            value_col=value_col,
            default_times=(48, 144),
            special_times={"7230cut": (72, 144)},
            style="bar",
        )
        safe_name = value_col.replace(" ", "_")
        save_figure(cov_fig, os.path.join(out_dir, f"cov_{safe_name}"))

        final_fig, _ = plot_final_size(
            df,
            value_col=value_col,
            final_time=144,
        )
        save_figure(final_fig, os.path.join(out_dir, f"final_size_{safe_name}"))

    plt.show()


if __name__ == "__main__":
    main()
