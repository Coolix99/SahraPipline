from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import scalar_path

DATA_DIR = scalar_path
WT_CSV = os.path.join(DATA_DIR, "membrane_dynamics_FINAL_200726_updated.csv")
OUT_DIR = Path("./wt_explorative_plots")

COLORS = {
    "Development": "#2077b5",
    "Regeneration": "#f57e1f",
    "4850cut": "#5b3a2a",
    "7230cut": "#d8b07a",
}
LABELS = {
    "Development": "Development",
    "Regeneration": "Regeneration",
    "4850cut": "Regeneration 50%",
    "7230cut": "Late amputation 30%",
}
XTICKS = [48, 60, 72, 84, 96, 120, 126, 132, 144]

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "DejaVu Sans",
})


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"condition", "time in hpf", "Surface Area", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    for column in ["time in hpf", "Surface Area", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["condition"].isin(COLORS)].copy()
    df["Mean thickness"] = df["Volume"] / df["Surface Area"]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def save_figure(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(OUT_DIR / f"{name}.{suffix}", bbox_inches="tight", **kwargs)


def style_axis(ax, ylabel: str):
    ax.set_xlim(45, 147)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XTICKS, rotation=45, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel("Developmental time [hpf]", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(XTICKS[0], XTICKS[-1])
    ax.legend(frameon=False, fontsize=14)
    plt.subplots_adjust(left=0.15, right=0.97, bottom=0.2025, top=0.979)


def summary_by_time(df, value_col):
    return (
        df.groupby("time in hpf")[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("time in hpf")
    )


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min((len(order) - rank) * p[index], 1.0))
        adjusted[index] = running
    return adjusted


def p_to_stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def timewise_test(df, value_col, first="Development", second="Regeneration"):
    common = sorted(
        set(df.loc[df["condition"] == first, "time in hpf"].dropna())
        & set(df.loc[df["condition"] == second, "time in hpf"].dropna())
    )
    rows = []
    for time in common:
        x = df.loc[(df["condition"] == first) & (df["time in hpf"] == time), value_col].dropna()
        y = df.loc[(df["condition"] == second) & (df["time in hpf"] == time), value_col].dropna()
        statistic, p_value = mannwhitneyu(x, y, alternative="two-sided")
        rows.append({
            "time in hpf": time,
            "n_development": len(x),
            "n_regeneration": len(y),
            "U": statistic,
            "p": p_value,
        })
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p"])
    result["significance"] = result["p_holm"].map(p_to_stars)
    return result


def add_raw_points(ax, sub, value_col, color):
    x = sub["time in hpf"].to_numpy(dtype=float)
    ax.scatter(
        x, sub[value_col], color=color, s=38, alpha=0.5,
        edgecolors="none", linewidths=0, zorder=1,
    )


def plot_thickness_dev_reg(df):
    conditions = ["Development", "Regeneration"]
    data = df[df["condition"].isin(conditions)].dropna(
        subset=["time in hpf", "Mean thickness"]
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for index, condition in enumerate(conditions):
        sub = data[data["condition"] == condition]
        stats = summary_by_time(sub, "Mean thickness")
        time = stats["time in hpf"].to_numpy()
        mean = stats["mean"].to_numpy()
        std = stats["std"].fillna(0).to_numpy()
        add_raw_points(ax, sub, "Mean thickness", COLORS[condition])
        ax.fill_between(
            time, mean - std, mean + std,
            color=COLORS[condition], alpha=0.2, linewidth=0, zorder=0,
        )
        ax.plot(time, mean, color=COLORS[condition], linewidth=2, zorder=2)
        ax.scatter(
            time, mean, color=COLORS[condition], edgecolor="white",
            linewidth=1.2, s=120, zorder=3, label=LABELS[condition],
        )

    tests = timewise_test(data, "Mean thickness")
    value_span = data["Mean thickness"].max() - data["Mean thickness"].min()
    value_span = value_span if value_span > 0 else 1.0
    for row in tests.itertuples(index=False):
        if row.significance == "ns":
            continue
        values = data.loc[data["time in hpf"] == row[0], "Mean thickness"]
        ax.text(
            row[0], values.max() + 0.035 * value_span, row.significance,
            ha="center", va="bottom", fontsize=16, color="black",
        )
    ax.margins(y=0.12)
    style_axis(ax, r"Mean thickness $\bar{h}$ [$\mu$m]")
    return fig, tests


def plot_regeneration_cuts(df, value_col, scale, ylabel, filename):
    conditions = ["Regeneration", "4850cut", "7230cut"]
    data = df[df["condition"].isin(conditions)].dropna(
        subset=["time in hpf", value_col]
    ).copy()
    data["_plot_value"] = data[value_col] / scale
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for index, condition in enumerate(conditions):
        sub = data[data["condition"] == condition]
        stats = summary_by_time(sub, "_plot_value")
        time = stats["time in hpf"].to_numpy()
        mean = stats["mean"].to_numpy()
        std = stats["std"].fillna(0).to_numpy()
        add_raw_points(ax, sub, "_plot_value", COLORS[condition])

        if condition == "Regeneration":
            ax.fill_between(
                time, mean - std, mean + std,
                color=COLORS[condition], alpha=0.2, linewidth=0, zorder=0,
            )
            ax.plot(
                time, mean, color=COLORS[condition], linewidth=2,
                zorder=2, label=LABELS[condition],
            )
        else:
            ax.errorbar(
                time, mean, yerr=std, fmt="none", ecolor=COLORS[condition],
                elinewidth=1.8, capsize=5, capthick=1.8, zorder=2,
            )

        ax.scatter(
            time, mean, color=COLORS[condition], edgecolor="white",
            linewidth=1.2, s=120, zorder=3,
            label=LABELS[condition] if condition != "Regeneration" else None,
        )

    style_axis(ax, ylabel)
    save_figure(fig, filename)
    return fig


def main():
    df = load_data(WT_CSV)
    print("\n--- Total sample sizes ---")
    for condition in ["Development", "Regeneration", "4850cut", "7230cut"]:
        n = int(df.loc[df["condition"] == condition, ["time in hpf", "Surface Area", "Volume"]].dropna().shape[0])
        print(f"{LABELS[condition]}: total n = {n}")
    thickness_fig, tests = plot_thickness_dev_reg(df)
    save_figure(thickness_fig, "mean_thickness_development_vs_regeneration")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tests.to_csv(OUT_DIR / "mean_thickness_timewise_tests.csv", index=False)
    print("\nDevelopment vs regeneration, mean thickness")
    print(tests.to_string(index=False, float_format=lambda value: f"{value:.4g}"))

    plot_regeneration_cuts(
        df, "Volume", 1e6,
        r"Volume [$10^6\,\mu$m$^3$]",
        "regeneration_cuts_volume",
    )
    plot_regeneration_cuts(
        df, "Surface Area", 1e4,
        r"Surface area [$(100\,\mu$m)$^2$]",
        "regeneration_cuts_surface_area",
    )
    plot_regeneration_cuts(
        df, "Mean thickness", 1.0,
        r"Mean thickness $\bar{h}$ [$\mu$m]",
        "regeneration_cuts_mean_thickness",
    )
    print(f"\nPlots and statistics written to: {OUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
