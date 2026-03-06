import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from matplotlib.ticker import MultipleLocator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import scalar_path


# =========================
# Global style / export
# =========================
mpl.rcParams.update({
    "svg.fonttype": "none",        # keep text editable in Illustrator
    "pdf.fonttype": 42,
    "text.usetex": False,
    "axes.unicode_minus": False
})

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


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
# Colors / labels
# =========================
COND_ORDER = ["Development", "Regeneration", "4850cut", "7230cut"]

COND_LABEL = {
    "Development": "Development",
    "Regeneration": "Regeneration 30%",
    "4850cut": "Regeneration 50%",
    "7230cut": "Late Amputation 30%",
}

COND_COLOR = {
    "Development": "#2278b5",
    "Regeneration": "#f57f20",
    "4850cut": "#017c91",
    "7230cut": "#b29dcb",
}


# =========================
# Helpers
# =========================
def clean_limits(vmin, vmax, step):
    vmin = step * np.floor(vmin / step)
    vmax = step * np.ceil(vmax / step)
    return vmin, vmax


def set_clean_yaxis(ax, ymin=None, ymax=None, step=None):
    cur_ymin, cur_ymax = ax.get_ylim()

    if ymin is None:
        ymin = cur_ymin
    if ymax is None:
        ymax = cur_ymax

    if step is None:
        span = ymax - ymin
        if span <= 0:
            step = 1.0
        else:
            rough = span / 5.0
            p = 10 ** np.floor(np.log10(rough))
            for m in [1, 2, 5, 10]:
                if m * p >= rough:
                    step = m * p
                    break

    ymin, ymax = clean_limits(ymin, ymax, step)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(step))


def set_clean_xaxis(ax, xmin=None, xmax=None, step=None):
    cur_xmin, cur_xmax = ax.get_xlim()

    if xmin is None:
        xmin = cur_xmin
    if xmax is None:
        xmax = cur_xmax

    if step is None:
        span = xmax - xmin
        if span <= 0:
            step = 1.0
        else:
            rough = span / 5.0
            p = 10 ** np.floor(np.log10(rough))
            for m in [1, 2, 5, 10]:
                if m * p >= rough:
                    step = m * p
                    break

    xmin, xmax = clean_limits(xmin, xmax, step)
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(step))


def save_figure(fig, filename_base, dpi=300):
    """
    Save figure as SVG, PDF, and PNG.

    filename_base should NOT include extension.
    """
    fig.savefig(f"{filename_base}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{filename_base}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{filename_base}.png", format="png", dpi=dpi, bbox_inches="tight")

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

    return cov(x), np.nanstd(boots, ddof=1)


def summarize_cov_by_groups(df, value_col, group_cols, n_boot=5000, seed=0):
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if isinstance(key, tuple):
            key_tuple = key
        else:
            key_tuple = (key,)

        x = g[value_col].to_numpy(dtype=float)
        c, c_sem = bootstrap_cov_sem(x, n_boot=n_boot, seed=seed)
        rows.append(key_tuple + (c, c_sem, len(g)))

    return pd.DataFrame(rows, columns=[*group_cols, "cov", "cov_sem", "n"])


# =========================
# Plot CoV grouped
# =========================
def plot_cov_grouped(
    df,
    value_col="Surface Area",
    time_col="time in hpf",
    condition_col="condition",
    default_times=(48, 144),
    special_times={"7230cut": (72, 144)},
    conditions=COND_ORDER,
    n_boot=5000,
    seed=0,
    fig_size=None,
    style="bar",
    gap=0.8,
    within=0.9,
):
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    if fig_size is None:
        fig_size = (5.6, 3.2) if style == "point" else (5.0, 3.6)

    fig, ax = plt.subplots(figsize=fig_size)

    x_pos, heights, yerr, colors = [], [], [], []
    cond_centers, xticklabels = [], []

    x0 = 0.0

    for cnd in conditions:
        times = special_times.get(cnd, default_times)

        sub = df[(df[condition_col] == cnd) & (df[time_col].isin(times))]
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

    if style == "bar":
        ax.bar(
            x_pos, heights,
            color=colors,
            width=0.75,
            edgecolor="none",
            zorder=2
        )
    else:
        ax.scatter(
            x_pos, heights,
            color=colors,
            s=35,
            zorder=3
        )

    ax.errorbar(
        x_pos, heights, yerr=yerr,
        fmt="none",
        ecolor="k",
        elinewidth=1.2,
        capsize=3,
        zorder=4
    )

    ax.set_ylabel(f"CoV ({'A' if value_col == 'Surface Area' else 'V'})")
    ax.set_title(f"{value_col} variability")

    ax.set_xticks(cond_centers)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    set_clean_yaxis(ax, ymin=0)

    sns.despine()
    ax.grid(False)
    fig.tight_layout()
    return fig, ax


# =========================
# Plot mean at one time
# =========================
def plot_mean_144(
    df,
    value_col="Surface Area",
    time_col="time in hpf",
    condition_col="condition",
    time=144,
    conditions=COND_ORDER,
    fig_size=(3.8, 3.8),
    style="bar",
    seed=0,
):
    sub = df[(df[time_col] == time) & (df[condition_col].isin(conditions))].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")

    scale = 1e4 if value_col == "Surface Area" else 1.0

    rows = []
    raw = {}

    for cnd in conditions:
        x = sub.loc[sub[condition_col] == cnd, value_col].to_numpy(dtype=float)
        x = x[np.isfinite(x)] / scale
        if len(x) == 0:
            continue

        mean = np.mean(x)
        sem = np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        rows.append((cnd, mean, sem))
        raw[cnd] = x

    summ = pd.DataFrame(rows, columns=["condition", "mean", "sem"])

    fig, ax = plt.subplots(figsize=fig_size)

    x_pos = np.arange(len(summ))
    heights = summ["mean"].to_numpy()
    yerr = summ["sem"].to_numpy()
    colors = [COND_COLOR[c] for c in summ["condition"]]
    labels = [COND_LABEL[c] for c in summ["condition"]]

    if style == "bar":
        ax.bar(
            x_pos, heights,
            color=colors,
            width=0.75,
            edgecolor="none",
            zorder=2
        )
        ax.errorbar(
            x_pos, heights, yerr=yerr,
            fmt="none",
            ecolor="k",
            elinewidth=1.2,
            capsize=3,
            zorder=4
        )

    else:
        rng = np.random.default_rng(seed)

        box_width = 0.34
        cap_width = 0.14

        for i, cnd in enumerate(summ["condition"]):
            x = raw[cnd]
            color = COND_COLOR[cnd]

            q2_3, q15_9, q50, q84_1, q97_7 = np.percentile(
                x, [2.3, 15.9, 50.0, 84.1, 97.7]
            )

            # individual data points
            jitter = rng.uniform(-0.10, 0.10, size=len(x))
            

            # box: +/- 1 sigma percentiles
            rect = plt.Rectangle(
                (x_pos[i] - box_width / 2, q15_9),
                box_width,
                q84_1 - q15_9,
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                zorder=0
            )
            ax.add_patch(rect)

            # median line
            ax.plot(
                [x_pos[i] - box_width / 2, x_pos[i] + box_width / 2],
                [q50, q50],
                color=color,
                linewidth=2.0,
                zorder=4
            )

            # whiskers: +/- 2 sigma percentiles
            ax.plot([x_pos[i], x_pos[i]], [q2_3, q15_9], color=color, linewidth=1.5, zorder=3)
            ax.plot([x_pos[i], x_pos[i]], [q84_1, q97_7], color=color, linewidth=1.5, zorder=3)

            # whisker caps
            ax.plot([x_pos[i] - cap_width / 2, x_pos[i] + cap_width / 2], [q2_3, q2_3],
                    color=color, linewidth=1.5, zorder=3)
            ax.plot([x_pos[i] - cap_width / 2, x_pos[i] + cap_width / 2], [q97_7, q97_7],
                    color=color, linewidth=1.5, zorder=3)
            ax.scatter(
                    np.full(len(x), x_pos[i]) + jitter,
                    x,
                    s=24,
                    color=color,
                    edgecolors="none",
                    zorder=2,
                    alpha=0.6
                        )
    if value_col == "Surface Area":
        ax.set_ylabel(r"A [$\,(100\,\mu m)^2$]")
    else:
        ax.set_ylabel(r"V [$\mu m^3$]")

    ax.set_title(f"{value_col} at {time} hpf")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    set_clean_yaxis(ax, ymin=0)

    sns.despine()
    ax.grid(False)
    fig.tight_layout()
    return fig, ax


# =========================
# IO
# =========================
def load_growth_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "time in hpf" in df.columns:
        df["time in hpf"] = pd.to_numeric(df["time in hpf"], errors="coerce")
    return df


# =========================
# Main
# =========================
def main():
    set_plot_style_big()

    csv_file = os.path.join(scalar_path, "scalarGrowthData_meshBased.csv")
    df = load_growth_csv(csv_file)

    print("\nAvailable times:")
    print(sorted(df["time in hpf"].dropna().unique()))

    print("\nAvailable conditions:")
    print(sorted(df["condition"].dropna().unique()))

    out_dir = os.path.join('./', "plots_cov_mean")
    os.makedirs(out_dir, exist_ok=True)

    for value_col in ["Volume", "Surface Area"]:
        fig1, ax1 = plot_cov_grouped(
            df,
            value_col=value_col,
            default_times=(48, 144),
            special_times={"7230cut": (72, 144)},
            style="bar",
        )
        save_figure(fig1, os.path.join(out_dir, f"cov_{value_col.replace(' ', '_')}"))

        fig2, ax2 = plot_mean_144(
            df,
            value_col=value_col,
            style="point",
        )
        save_figure(fig2, os.path.join(out_dir, f"mean144_{value_col.replace(' ', '_')}"))

    plt.show()


if __name__ == "__main__":
    main()