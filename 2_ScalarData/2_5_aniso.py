#!/usr/bin/env python3
"""
growth_anisotropy.py

Determine growth anisotropy from mesh-based scalars.

- log-log scatter of L_PD_midline vs L_AP_40line
- global fit (all data): black dotted line + shaded band
- per-condition fits (Development=blue, Regeneration=orange)
- optional: extra fit for Regeneration restricted to hpf >= 72 (orange dashed)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# If you want to reuse your config scalar_path:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import scalar_path  # noqa: E402


# ============================================================
# Styling
# ============================================================

def set_plot_style():
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })


# ============================================================
# Helpers: filtering and regression in log-log
# ============================================================

def clean_xy(df, xcol, ycol, extra_mask=None):
    x = np.asarray(df[xcol], dtype=float)
    y = np.asarray(df[ycol], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if extra_mask is not None:
        mask &= np.asarray(extra_mask, dtype=bool)

    return x[mask], y[mask], mask


def fit_loglog_with_ci(x, y, xx, n_boot=2000, seed=0):
    """
    Fit log10(y) = a + b log10(x).
    Returns:
      yhat(xx), (ylo(xx), yhi(xx)), (a, b)
    Bootstrap CI is percentile band over predictions.
    """
    rng = np.random.default_rng(seed)

    lx = np.log10(x)
    ly = np.log10(y)

    # OLS in log space
    A = np.vstack([np.ones_like(lx), lx]).T
    a, b = np.linalg.lstsq(A, ly, rcond=None)[0]

    lxx = np.log10(xx)
    lyhat = a + b * lxx
    yhat = 10 ** lyhat

    # bootstrap prediction bands
    if len(x) < 3:
        return yhat, (yhat, yhat), (a, b)

    preds = np.empty((n_boot, len(xx)), dtype=float)
    n = len(x)
    idx = np.arange(n)

    for i in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        lx_s = lx[samp]
        ly_s = ly[samp]
        A_s = np.vstack([np.ones_like(lx_s), lx_s]).T
        a_s, b_s = np.linalg.lstsq(A_s, ly_s, rcond=None)[0]
        preds[i] = 10 ** (a_s + b_s * lxx)

    ylo = np.percentile(preds, 2.5, axis=0)
    yhi = np.percentile(preds, 97.5, axis=0)

    return yhat, (ylo, yhi), (a, b)


def format_fit_label(a, b):
    # y = 10^a * x^b
    c = 10 ** a
    return f"y = {c:.2g} · x^{b:.2f}"


# ============================================================
# Core plot
# ============================================================

def plot_anisotropy_loglog(
    df,
    xcol="L_PD_midline",
    ycol="L_AP_40line",
    condition_col="condition",
    time_col="time in hpf",
    conditions=("Development", "Regeneration"),
    regen_time_cut=72,
    title="Growth anisotropy (log-log)",
    fig_size=(3.4, 2.8),
    n_boot=2000,
    seed=0,
):
    """
    Scatter log-log of y vs x with:
      - global fit (all data): black dotted + shaded CI
      - per condition fits: blue (Development), orange (Regeneration) + shaded CI
      - extra fit for Regeneration with hpf>=regen_time_cut: orange dashed
    """
    fig = plt.figure(figsize=fig_size)
    ax = plt.gca()

    # Colors to match your previous convention
    palette = {
        "Development": sns.color_palette()[0],  # blue
        "Regeneration": sns.color_palette()[1],  # orange
    }

    # Determine plotting range from all valid data
    x_all, y_all, mask_all = clean_xy(df, xcol, ycol)
    if len(x_all) < 3:
        raise ValueError("Not enough valid positive data points for log-log plot.")

    xmin, xmax = np.min(x_all), np.max(x_all)
    xx = np.logspace(np.log10(xmin), np.log10(xmax), 200)

    # --- Scatter: per condition ---
    for cond in conditions:
        sub = df[df[condition_col] == cond]
        x, y, _ = clean_xy(sub, xcol, ycol)
        if len(x) == 0:
            continue
        ax.scatter(x, y, s=18, alpha=0.65, label=cond, color=palette.get(cond, None))

    # --- Global fit: all data (black dotted + shaded) ---
    yhat, (ylo, yhi), (a, b) = fit_loglog_with_ci(x_all, y_all, xx, n_boot=n_boot, seed=seed)
    ax.plot(xx, yhat, linestyle=":", linewidth=2, color="black", label=f"All: {format_fit_label(a,b)}")
    ax.fill_between(xx, ylo, yhi, color="black", alpha=0.12)

    # --- Per-condition fits (solid + shaded) ---
    for cond in conditions:
        sub = df[df[condition_col] == cond]
        x, y, _ = clean_xy(sub, xcol, ycol)
        if len(x) < 3:
            continue
        yhat_c, (ylo_c, yhi_c), (a_c, b_c) = fit_loglog_with_ci(
            x, y, xx, n_boot=n_boot, seed=seed + (1 if cond == "Development" else 2)
        )
        ax.plot(xx, yhat_c, linewidth=2, color=palette.get(cond, None),
                label=f"{cond}: {format_fit_label(a_c,b_c)}")
        ax.fill_between(xx, ylo_c, yhi_c, color=palette.get(cond, None), alpha=0.18)

    # --- Regeneration fit for late times only (hpf >= cut): dashed orange ---
    if "Regeneration" in conditions and time_col in df.columns:
        sub_reg = df[df[condition_col] == "Regeneration"].copy()
        late_mask = np.asarray(sub_reg[time_col], dtype=float) >= float(regen_time_cut)
        x_late, y_late, _ = clean_xy(sub_reg, xcol, ycol, extra_mask=late_mask)
        if len(x_late) >= 3:
            yhat_l, (ylo_l, yhi_l), (a_l, b_l) = fit_loglog_with_ci(
                x_late, y_late, xx, n_boot=n_boot, seed=seed + 99
            )
            ax.plot(xx, yhat_l, linestyle="--", linewidth=2,
                    color=palette["Regeneration"],
                    label=f"Regen ≥{regen_time_cut}hpf: {format_fit_label(a_l,b_l)}")
            ax.fill_between(xx, ylo_l, yhi_l, color=palette["Regeneration"], alpha=0.10)

    # --- Axes formatting ---
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L_PD_midline")
    ax.set_ylabel("L_AP_40line")
    ax.set_title(title)

    sns.despine()
    ax.grid(False)
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    return fig, ax


# ============================================================
# IO + Main
# ============================================================

def load_growth_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Make columns consistent with your example header
    # (keep as-is if already correct)
    # Ensure "time in hpf" numeric
    if "time in hpf" in df.columns:
        df["time in hpf"] = pd.to_numeric(df["time in hpf"], errors="coerce")

    return df


def main():
    set_plot_style()

    csv_file = os.path.join(scalar_path, "scalarGrowthData_meshBased.csv")
    df = load_growth_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")

    # Default: Development + Regeneration
    plot_anisotropy_loglog(
        df,
        xcol="L_PD_midline",
        ycol="L_AP_40line",
        condition_col="condition",
        time_col="time in hpf",
        conditions=("Development", "Regeneration"),
        regen_time_cut=84,
        title="Growth anisotropy: AP vs PD (log-log)",
        fig_size=(3.4, 2.8),
        n_boot=2000,
        seed=0,
    )

    # Panel label example (optional)
    ax = plt.gca()
    ax.text(-0.25, 1.15, "A",
            ha="left", va="top", weight="bold",
            transform=ax.transAxes, fontsize=14)

    plt.show()

    # Optional: include other conditions if present
    # Example:
    # unique_conds = tuple(sorted(df["condition"].dropna().unique()))
    # plot_anisotropy_loglog(df, conditions=unique_conds, title="All conditions")


if __name__ == "__main__":
    main()
