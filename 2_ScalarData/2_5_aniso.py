import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, FixedLocator


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
      yhat(xx), (ylo(xx), yhi(xx)), (a, b), (sa, sb)
    """
    rng = np.random.default_rng(seed)

    lx = np.log10(x)
    ly = np.log10(y)

    A = np.vstack([np.ones_like(lx), lx]).T
    coeffs, residuals, rank, svals = np.linalg.lstsq(A, ly, rcond=None)
    a, b = coeffs

    # --- standard errors ---
    n = len(lx)
    dof = n - 2
    sigma2 = residuals[0] / dof if dof > 0 else np.nan
    cov = sigma2 * np.linalg.inv(A.T @ A)
    sa, sb = np.sqrt(np.diag(cov))

    lxx = np.log10(xx)
    yhat = 10 ** (a + b * lxx)

    # --- bootstrap CI ---
    if n < 3:
        return yhat, (yhat, yhat), (a, b), (sa, sb)

    preds = np.empty((n_boot, len(xx)))
    idx = np.arange(n)

    for i in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        A_s = np.vstack([np.ones_like(lx[samp]), lx[samp]]).T
        a_s, b_s = np.linalg.lstsq(A_s, ly[samp], rcond=None)[0]
        preds[i] = 10 ** (a_s + b_s * lxx)

    ylo = np.percentile(preds, 2.5, axis=0)
    yhi = np.percentile(preds, 97.5, axis=0)

    # --- R^2 in log space ---
    ly_hat = a + b * lx
    ss_res = np.sum((ly - ly_hat) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


    return yhat, (ylo, yhi), (a, b), (sa, sb), r2



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

    # --- Scatter ---
    # Development
    sub_dev = df[df[condition_col] == "Development"]
    x, y, _ = clean_xy(sub_dev, xcol, ycol)
    ax.scatter(x, y, s=5, alpha=1.0,
            color=palette["Development"],
            label="Development")

    # Regeneration: early
    sub_reg = df[df[condition_col] == "Regeneration"]

    early_mask = sub_reg[time_col] < regen_time_cut
    x_e, y_e, _ = clean_xy(sub_reg, xcol, ycol, extra_mask=early_mask)
    ax.scatter(x_e, y_e, s=5, alpha=0.5,
            color=palette["Regeneration"],
            label=f"Regen <{regen_time_cut}hpf")

    # Regeneration: late
    late_mask = sub_reg[time_col] >= regen_time_cut
    x_l, y_l, _ = clean_xy(sub_reg, xcol, ycol, extra_mask=late_mask)
    ax.scatter(x_l, y_l, s=5, alpha=1.0,
            color=palette["Regeneration"],
            label=f"Regen ≥{regen_time_cut}hpf")


    # --- Per-condition fits (solid + shaded) ---
    for cond in conditions:
        sub = df[df[condition_col] == cond]

        if cond != "Regeneration":
            x, y, _ = clean_xy(sub, xcol, ycol)
            if len(x) < 3:
                continue

            yhat_c, (ylo_c, yhi_c), (a_c, b_c), (sa_c, sb_c), r2_c = \
                fit_loglog_with_ci(x, y, xx, n_boot=n_boot, seed=seed + 1)

            print(
                f"{cond}: "
                f"a = {a_c:.3f} ± {sa_c:.3f}, "
                f"b = {b_c:.3f} ± {sb_c:.3f}, "
                f"R² = {r2_c:.3f}"
            )

            ax.plot(xx, yhat_c, linewidth=2,
                    color=palette[cond],
                    label=f"{cond}: {format_fit_label(a_c,b_c)}")
            # ax.fill_between(xx, ylo_c, yhi_c,
            #                 color=palette[cond], alpha=0.18)

        else:
            
            # --- Regeneration: before cut ---
            early_mask = sub[time_col] < regen_time_cut
            x_e, y_e, _ = clean_xy(sub, xcol, ycol, extra_mask=early_mask)
            xmin, xmax = np.min(x_e), np.max(x_e)
            xx_e = np.logspace(np.log10(xmin), np.log10(xmax), 200)
            if len(x_e) >= 3:
                yhat_e, (ylo_e, yhi_e), (a_e, b_e), (sa_e, sb_e), r2_e = \
                    fit_loglog_with_ci(x_e, y_e, xx_e, n_boot=n_boot, seed=seed + 2)

                print(
                    f"Regeneration <{regen_time_cut} hpf: "
                    f"a = {a_e:.3f} ± {sa_e:.3f}, "
                    f"b = {b_e:.3f} ± {sb_e:.3f}, "
                    f"R² = {r2_e:.3f}"
                )

                ax.plot(xx_e, yhat_e, linewidth=2, linestyle="--",
                        color=palette["Regeneration"],
                        label=f"Regen <{regen_time_cut}hpf")
                # ax.fill_between(xx_e, ylo_e, yhi_e,
                #                 color=palette["Regeneration"], alpha=0.18)

            # --- Regeneration: after cut ---
            late_mask = sub[time_col] >= regen_time_cut
            x_l, y_l, _ = clean_xy(sub, xcol, ycol, extra_mask=late_mask)

            if len(x_l) >= 3:
                yhat_l, (ylo_l, yhi_l), (a_l, b_l), (sa_l, sb_l), r2_l = \
                    fit_loglog_with_ci(x_l, y_l, xx, n_boot=n_boot, seed=seed + 3)

                print(
                    f"Regeneration ≥{regen_time_cut} hpf: "
                    f"a = {a_l:.3f} ± {sa_l:.3f}, "
                    f"b = {b_l:.3f} ± {sb_l:.3f}, "
                    f"R² = {r2_l:.3f}"
                )

                ax.plot(xx, yhat_l, linewidth=2,
                        color=palette["Regeneration"],
                        label=f"Regen ≥{regen_time_cut}hpf")
                # ax.fill_between(xx, ylo_l, yhi_l,
                #                 color=palette["Regeneration"], alpha=0.10)


    # --- Axes formatting ---
    ax.set_xscale("log")
    ax.set_yscale("log")

    ticks = [100, 200, 300, 400, 500]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_locator(FixedLocator(ticks))

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlim(90, 590)
    ax.set_ylim(90, 590)

    ax.set_xlabel(r"PD Length [$\mu$m]")
    ax.set_ylabel(r"AP Length [$\mu$m]")
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
