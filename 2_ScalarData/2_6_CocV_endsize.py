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
COND_ORDER = ["Development", "Regeneration", "4850cut", "7230cut"]

COND_LABEL = {
    "Development": "Development",
    "Regeneration": "Regeneration 30%",
    "4850cut": "Regeneration 50%",
    "7230cut": "Late Amputation 30%",
}

COND_COLOR = {
    "Development": sns.color_palette()[0],      # blue
    "Regeneration": sns.color_palette()[1],     # orange
    "4850cut": "cyan",                          # changed
    "7230cut": "0.5",                           # grey
}

def plot_cov_grouped(
    df,
    value_col="Surface Area",
    time_col="time in hpf",
    condition_col="condition",
    default_times=(48, 144),
    special_times={"7230cut": (72, 144)},  # <-- key change
    conditions=COND_ORDER,
    n_boot=5000,
    seed=0,
    fig_size=(7.5, 3.6),
    style="bar",
    gap=0.8,
    within=0.9,
):
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    fig, ax = plt.subplots(figsize=fig_size)

    x_pos, heights, yerr, colors = [], [], [], []
    cond_centers, xticklabels = [], []

    x0 = 0.0

    for cnd in conditions:

        # --- choose which timepoints to compare ---
        times = special_times.get(cnd, default_times)

        sub = df[(df[condition_col] == cnd) &
                 (df[time_col].isin(times))]

        if len(sub) == 0:
            x0 += gap
            continue

        summ = summarize_cov_by_groups(
            sub,
            value_col=value_col,
            group_cols=[time_col],
            n_boot=n_boot,
            seed=seed
        )

        pos_list = []

        for i, t in enumerate(times):
            row = summ[summ[time_col] == t]
            if len(row) == 0:
                continue

            pos = x0 + i * within
            pos_list.append(pos)

            x_pos.append(pos)
            heights.append(float(row["cov"].iloc[0]))
            yerr.append(float(row["cov_sem"].iloc[0]))
            colors.append(COND_COLOR[cnd])

        if len(pos_list) > 0:
            cond_centers.append(np.mean(pos_list))
            xticklabels.append(COND_LABEL[cnd])
            x0 = pos_list[-1] + gap
        else:
            x0 += gap

    x_pos = np.array(x_pos)
    heights = np.array(heights)
    yerr = np.array(yerr)

    # ---- draw ----
    if style == "bar":
        ax.bar(x_pos, heights, color=colors, width=0.75,
               edgecolor="none", zorder=2)
    else:
        ax.scatter(x_pos, heights, color=colors, s=35, zorder=3)

    ax.errorbar(
        x_pos, heights, yerr=yerr,
        fmt="none", ecolor="k",
        elinewidth=1.5, capsize=3, zorder=4
    )

    ax.set_ylabel(f"CoV ({'A' if value_col=='Surface Area' else 'V'})")
    ax.set_title(f"{value_col} variability")

    ax.set_xticks(cond_centers)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    sns.despine()
    ax.grid(False)
    plt.tight_layout()
    return fig, ax
def plot_mean_144(
    df,
    value_col="Surface Area",   # or "Volume"
    time_col="time in hpf",
    condition_col="condition",
    time=144,
    conditions=COND_ORDER,
    fig_size=(4.5, 3.6),
    style="bar",
):
    sub = df[(df[time_col] == time) & (df[condition_col].isin(conditions))].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")

    scale = 1e4 if value_col == "Surface Area" else 1.0

    rows = []
    for cnd in conditions:
        x = sub.loc[sub[condition_col] == cnd, value_col].to_numpy(dtype=float)
        x = x[np.isfinite(x)] / scale
        n = len(x)
        if n == 0:
            continue
        mean = np.mean(x)
        sem = np.std(x, ddof=1)/np.sqrt(n) if n > 1 else np.nan
        rows.append((cnd, mean, sem))

    summ = pd.DataFrame(rows, columns=["condition", "mean", "sem"])

    fig, ax = plt.subplots(figsize=fig_size)

    x_pos = np.arange(len(summ))
    heights = summ["mean"].to_numpy()
    yerr = summ["sem"].to_numpy()
    colors = [COND_COLOR[c] for c in summ["condition"]]
    labels = [COND_LABEL[c] for c in summ["condition"]]

    if style == "bar":
        ax.bar(x_pos, heights, color=colors, width=0.75, edgecolor="none", zorder=2)
    else:
        ax.scatter(x_pos, heights, color=colors, s=40, zorder=3)

    ax.errorbar(
        x_pos, heights, yerr=yerr,
        fmt="none", ecolor="k", elinewidth=1.5, capsize=3, zorder=4
    )

    if value_col == "Surface Area":
        ax.set_ylabel(r"A [$\,(100\,\mu m)^2$]")
    else:
        ax.set_ylabel("Volume")

    ax.set_title(f"{value_col} at {time} hpf")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    sns.despine()
    ax.grid(False)
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
    set_plot_style_big()

    csv_file = os.path.join(scalar_path, "scalarGrowthData_meshBased.csv")
    df = load_growth_csv(csv_file)

    print("\nAvailable times:")
    print(sorted(df["time in hpf"].dropna().unique()))

    print("\nAvailable conditions:")
    print(sorted(df["condition"].dropna().unique()))

    # --- CoV plot ---
    fig1, ax1 = plot_cov_grouped(
        df,
        value_col="Volume",#"Surface Area",   # or 
        default_times=(48, 144),    # optional (this is already default)
        special_times={"7230cut": (72, 144)},  # optional (already default)
        style="bar",                # or "point"
    )

    # --- Mean size plot ---
    fig2, ax2 = plot_mean_144(
        df,
        value_col="Volume",#"Surface Area",   # or 
        style="bar",
    )

    plt.show()



if __name__ == "__main__":
    main()
