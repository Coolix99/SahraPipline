import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.ticker import ScalarFormatter, FixedLocator 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from config import scalar_path
# =========================
def set_plot_style_big():
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

# =========================
# CoV + bootstrap SEM
# =========================
def cov(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    m = np.mean(x)
    if m == 0:
        return np.nan
    return np.std(x, ddof=1) / m

def bootstrap_cov_sem(x, n_boot=5000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return cov(x), np.nan

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for i in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        boots[i] = cov(x[samp])

    return cov(x), np.nanstd(boots, ddof=1)  # SEM estimate for CoV

def summarize_cov_by_groups(df, value_col, group_cols, n_boot=5000, seed=0):
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        # key is a scalar if len(group_cols)==1, otherwise a tuple
        if isinstance(key, tuple):
            key_tuple = key
        else:
            key_tuple = (key,)

        x = g[value_col].to_numpy(dtype=float)
        c, c_sem = bootstrap_cov_sem(x, n_boot=n_boot, seed=seed)

        rows.append(key_tuple + (c, c_sem, len(g)))

    out = pd.DataFrame(rows, columns=[*group_cols, "cov", "cov_sem", "n"])
    return out

# =========================
# Condition config
# =========================
COND_ORDER = ["Development", "Regeneration", "4850cut"]
COND_LABEL = {
    "Development": "Development",
    "Regeneration": "Regeneration 30%",
    "4850cut": "Regeneration 50%",
}
COND_COLOR = {
    "Development": sns.color_palette()[0],  # blue
    "Regeneration": sns.color_palette()[1],  # orange
    "4850cut": sns.color_palette()[3],       # red-ish (tab:red-ish in seaborn)
}

# =========================
# Plot 1: CoV at 48 vs 144 (grouped by condition)
# =========================
def plot_cov_surfacearea_48_144_grouped(
    df,
    value_col="Surface Area",
    time_col="time in hpf",
    condition_col="condition",
    times=(48, 144),
    conditions=("Development", "Regeneration", "4850cut"),
    n_boot=5000,
    seed=0,
    fig_size=(6.6, 3.4),
    style="bar",          # "bar" or "point"
    gap=0.8,              # gap between condition groups
    within=0.9,           # spacing between 48 and 144 within group
):
    sub = df[df[time_col].isin(times) & df[condition_col].isin(conditions)].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")

    summ = summarize_cov_by_groups(
        sub, value_col=value_col,
        group_cols=[time_col, condition_col],
        n_boot=n_boot, seed=seed
    )

    # order
    summ[time_col] = pd.Categorical(summ[time_col], categories=list(times), ordered=True)
    summ[condition_col] = pd.Categorical(summ[condition_col], categories=list(conditions), ordered=True)
    summ = summ.sort_values([condition_col, time_col])

    fig, ax = plt.subplots(figsize=fig_size)

    # positions: for each condition -> two adjacent positions for 48/144, then a gap
    x_pos = []
    heights = []
    yerr = []
    colors = []

    cond_centers = []
    xticklabels = []

    x0 = 0.0
    for cnd in conditions:
        # positions for (48,144)
        pos_48 = x0
        pos_144 = x0 + within
        cond_centers.append((pos_48 + pos_144) / 2.0)
        xticklabels.append(COND_LABEL.get(cnd, str(cnd)))

        for t, pos in zip(times, [pos_48, pos_144]):
            row = summ[(summ[condition_col] == cnd) & (summ[time_col] == t)]
            x_pos.append(pos)

            if len(row) == 1:
                heights.append(float(row["cov"].iloc[0]))
                yerr.append(float(row["cov_sem"].iloc[0]))
            else:
                heights.append(np.nan)
                yerr.append(np.nan)

            colors.append(COND_COLOR.get(cnd, "0.5"))

        # advance to next group
        x0 = x0 + within + gap

    x_pos = np.array(x_pos, float)
    heights = np.array(heights, float)
    yerr = np.array(yerr, float)

    if style == "bar":
        ax.bar(x_pos, heights, color=colors, width=0.75, edgecolor="none", zorder=2)
    elif style == "point":
        ax.scatter(x_pos, heights, color=colors, s=30, zorder=3)
    else:
        raise ValueError("style must be 'bar' or 'point'")

    # black error bars always
    ax.errorbar(
        x_pos, heights, yerr=yerr,
        fmt="none", ecolor="k", elinewidth=1.5, capsize=3, zorder=4
    )

    ax.set_ylabel("CoV (A)")
    ax.set_title("Surface Area variability (CoV): 48 vs 144 hpf")

    # one label per condition group
    ax.set_xticks(cond_centers)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    sns.despine()
    ax.grid(False)
    plt.tight_layout()
    return fig, ax, summ


# =========================
# Plot 2: mean Surface Area at 144 with SEM (scaled)
# =========================
def plot_surfacearea_mean_144_scaled(
    df,
    value_col="Surface Area",
    time_col="time in hpf",
    condition_col="condition",
    time=144,
    conditions=("Development", "Regeneration", "4850cut"),
    fig_size=(3.8, 3.4),
    style="bar",  # "bar" or "point"
    scale=1e4,
):
    sub = df[(df[time_col] == time) & (df[condition_col].isin(conditions))].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")

    rows = []
    for cnd in conditions:
        x = sub.loc[sub[condition_col] == cnd, value_col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        x = x / scale  # scale here
        n = len(x)
        m = np.nanmean(x) if n > 0 else np.nan
        sem = (np.nanstd(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        rows.append((cnd, m, sem, n))

    summ = pd.DataFrame(rows, columns=[condition_col, "mean", "sem", "n"])

    fig, ax = plt.subplots(figsize=fig_size)

    x_pos = np.arange(len(conditions))
    heights = summ["mean"].to_numpy(float)
    yerr = summ["sem"].to_numpy(float)
    colors = [COND_COLOR.get(c, "0.5") for c in conditions]
    labels = [COND_LABEL.get(c, str(c)) for c in conditions]

    if style == "bar":
        ax.bar(x_pos, heights, color=colors, width=0.75, edgecolor="none", zorder=2)
    elif style == "point":
        ax.scatter(x_pos, heights, color=colors, s=40, zorder=3)
    else:
        raise ValueError("style must be 'bar' or 'point'")

    ax.errorbar(
        x_pos, heights, yerr=yerr,
        fmt="none", ecolor="k", elinewidth=1.5, capsize=3, zorder=4
    )

    ax.set_ylabel(r"A [$\,(100\,\mu m)^2$]")
    ax.set_title("Surface Area at 144 hpf")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    sns.despine()
    ax.grid(False)
    plt.tight_layout()
    return fig, ax, summ
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
    set_plot_style_big()
    csv_file = os.path.join(scalar_path, "scalarGrowthData_meshBased.csv")
    df = load_growth_csv(csv_file)
    fig1, ax1, cov_summ = plot_cov_surfacearea_48_144_grouped(
        df,
        times=(48, 144),
        conditions=("Development", "Regeneration", "4850cut"),
        style="bar",   # or "point"
    )

    fig2, ax2, mean_summ = plot_surfacearea_mean_144_scaled(
        df,
        time=144,
        conditions=("Development", "Regeneration", "4850cut"),
        style="bar",   # or "point"
    )

    plt.show()



if __name__ == "__main__":
    main()
