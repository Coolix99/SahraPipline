import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import scalar_path


# ============================================================
# Fit helpers
# ============================================================

def fit_mx(x, y):
    return np.sum(x * y) / np.sum(x * x)


def fit_mx_n(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    m, n = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, n


def scatter_with_fits(x, y, xlabel, ylabel, title):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, alpha=0.7)

    xx = np.linspace(x.min(), x.max(), 200)

    m = fit_mx(x, y)
    plt.plot(xx, m * xx, linestyle=":", linewidth=2, color="black",
             label=f"y = {m:.2f} x")

    m2, n2 = fit_mx_n(x, y)
    plt.plot(xx, m2 * xx + n2, linestyle="--", linewidth=2, color="black",
             label=f"y = {m2:.2f} x + {n2:.2f}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# Axis analysis
# ============================================================

def analyze_axes(df):
    # -------------------------
    # Scatter plots
    # -------------------------
    scatter_with_fits(
        df["L_PD_BB"], df["L_PD_midline"],
        xlabel="L_PD_BB",
        ylabel="L_PD_midline",
        title="PD bounding box vs midline"
    )

    scatter_with_fits(
        df["L_AP_BB"], df["L_AP_40line"],
        xlabel="L_AP_BB",
        ylabel="L_AP_40line",
        title="AP bounding box vs AP @ 40% PD"
    )

    scatter_with_fits(
        df["L_AP_BB"], df["L_AP_longline"],
        xlabel="L_AP_BB",
        ylabel="L_AP_longline",
        title="AP bounding box vs longest AP"
    )

    scatter_with_fits(
        df["L_AP_40line"], df["L_AP_longline"],
        xlabel="L_AP_40line",
        ylabel="L_AP_longline",
        title="AP @ 40% PD vs longest AP"
    )

    # -------------------------
    # Histograms
    # -------------------------
    plt.figure(figsize=(5, 4))
    plt.hist(df["PD_long_rel"].dropna(), bins=20)
    plt.xlabel("PD_long_rel")
    plt.ylabel("Count")
    plt.title("Relative PD position of longest AP line")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 4))
    plt.hist(df["L_DV_npts"].dropna(), bins=20)
    plt.xlabel("Number of thickness points")
    plt.ylabel("Count")
    plt.title("Thickness sampling point count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# Group statistics & variability
# ============================================================

def analyze_variability(df):
    group_cols = ["genotype", "time in hpf", "condition"]
    grouped = df.groupby(group_cols)

    stats = grouped.agg(
        # PD
        mean_PD_BB=("L_PD_BB", "mean"),
        std_PD_BB=("L_PD_BB", "std"),
        mean_PD_mid=("L_PD_midline", "mean"),
        std_PD_mid=("L_PD_midline", "std"),

        # AP
        mean_AP_BB=("L_AP_BB", "mean"),
        std_AP_BB=("L_AP_BB", "std"),
        mean_AP_40=("L_AP_40line", "mean"),
        std_AP_40=("L_AP_40line", "std"),
        mean_AP_long=("L_AP_longline", "mean"),
        std_AP_long=("L_AP_longline", "std"),

        n=("L_PD_BB", "count"),
    )

    # Coefficient of variation
    stats["cv_PD_BB"] = stats["std_PD_BB"] / stats["mean_PD_BB"]
    stats["cv_PD_mid"] = stats["std_PD_mid"] / stats["mean_PD_mid"]

    stats["cv_AP_BB"] = stats["std_AP_BB"] / stats["mean_AP_BB"]
    stats["cv_AP_40"] = stats["std_AP_40"] / stats["mean_AP_40"]
    stats["cv_AP_long"] = stats["std_AP_long"] / stats["mean_AP_long"]

    print("\n===== Group statistics (axis only) =====\n")
    print(stats)

    # -------------------------
    # Variability plots
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # PD std
    x_pd = [0, 1]
    labels_pd = ["L_PD_BB", "L_PD_midline"]
    for _, r in stats.iterrows():
        axes[0, 0].plot(x_pd, [r["std_PD_BB"], r["std_PD_mid"]],
                        marker="o", alpha=0.6)

    axes[0, 0].scatter(
        x_pd,
        [stats["std_PD_BB"].mean(), stats["std_PD_mid"].mean()],
        color="red", marker="x", s=100, label="Mean"
    )
    axes[0, 0].set_xticks(x_pd)
    axes[0, 0].set_xticklabels(labels_pd)
    axes[0, 0].set_ylabel("Standard deviation")
    axes[0, 0].set_title("PD variability (std)")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # PD CoV
    for _, r in stats.iterrows():
        axes[0, 1].plot(x_pd, [r["cv_PD_BB"], r["cv_PD_mid"]],
                        marker="o", alpha=0.6)

    axes[0, 1].scatter(
        x_pd,
        [stats["cv_PD_BB"].mean(), stats["cv_PD_mid"].mean()],
        color="red", marker="x", s=100, label="Mean"
    )
    axes[0, 1].set_xticks(x_pd)
    axes[0, 1].set_xticklabels(labels_pd)
    axes[0, 1].set_ylabel("CoV")
    axes[0, 1].set_title("PD variability (CoV)")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # AP std
    x_ap = [0, 1, 2]
    labels_ap = ["L_AP_BB", "L_AP_40line", "L_AP_longline"]
    for _, r in stats.iterrows():
        axes[1, 0].plot(
            x_ap,
            [r["std_AP_BB"], r["std_AP_40"], r["std_AP_long"]],
            marker="o", alpha=0.6
        )

    axes[1, 0].scatter(
        x_ap,
        [
            stats["std_AP_BB"].mean(),
            stats["std_AP_40"].mean(),
            stats["std_AP_long"].mean()
        ],
        color="red", marker="x", s=100, label="Mean"
    )
    axes[1, 0].set_xticks(x_ap)
    axes[1, 0].set_xticklabels(labels_ap)
    axes[1, 0].set_ylabel("Standard deviation")
    axes[1, 0].set_title("AP variability (std)")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # AP CoV
    for _, r in stats.iterrows():
        axes[1, 1].plot(
            x_ap,
            [r["cv_AP_BB"], r["cv_AP_40"], r["cv_AP_long"]],
            marker="o", alpha=0.6
        )

    axes[1, 1].scatter(
        x_ap,
        [
            stats["cv_AP_BB"].mean(),
            stats["cv_AP_40"].mean(),
            stats["cv_AP_long"].mean()
        ],
        color="red", marker="x", s=100, label="Mean"
    )
    axes[1, 1].set_xticks(x_ap)
    axes[1, 1].set_xticklabels(labels_ap)
    axes[1, 1].set_ylabel("CoV")
    axes[1, 1].set_title("AP variability (CoV)")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def analyze_L_PD_S(df_joined):
    """
    Compare mesh-based PD measures with annotation-based PD measures.
    """

    df = df_joined.copy()

    # -------------------------
    # Annotation-based PD measures
    # -------------------------
    df["L_PD_S_dx"] = np.abs(df["p2_x"] - df["p1_x"])
    df["L_PD_S_dxdy"] = np.sqrt(
        (df["p2_x"] - df["p1_x"])**2 +
        (df["p2_y"] - df["p1_y"])**2
    )

    print("\nAdded annotation-based PD measures:")
    print(df[["L_PD_S_dx", "L_PD_S_dxdy"]].head())

    # -------------------------
    # Scatter comparisons
    # -------------------------
    scatter_with_fits(
        df["L_PD_S_dxdy"], df["L_PD_S_dx"],
        xlabel="L_PD_S_dxdy (annotation)",
        ylabel="L_PD_S_dx (annotation)",
        title="PD: annotation distancevs annotation Δx"
    )
    # -------------------------
    # Annotation line angle
    # -------------------------
    df["angle_deg"] = np.degrees(
        np.arctan2(
            df["p2_y"] - df["p1_y"],
            np.abs(df["p2_x"] - df["p1_x"])
        )
    )

    print("\nAnnotation line angle (deg) summary:")
    print(df["angle_deg"].describe())

    # -------------------------
    # Histogram of angles
    # -------------------------
    plt.figure(figsize=(5, 4))
    plt.hist(
        df["angle_deg"],
        bins=30,
        range=(-15, 15)
    )
    plt.xlabel("Annotation line angle (degrees)")
    plt.ylabel("Count")
    plt.title("Orientation of annotated PD lines")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    scatter_with_fits(
        df["L_PD_BB"], df["L_PD_S_dx"],
        xlabel="L_PD_BB (mesh)",
        ylabel="L_PD_S_dx (annotation)",
        title="PD: Bounding box vs annotation Δx"
    )

    scatter_with_fits(
        df["L_PD_midline"], df["L_PD_S_dx"],
        xlabel="L_PD_midline (mesh)",
        ylabel="L_PD_S_dx (annotation)",
        title="PD: Midline vs annotation Δx"
    )

    scatter_with_fits(
        df["L_PD_BB"], df["L_PD_S_dxdy"],
        xlabel="L_PD_BB (mesh)",
        ylabel="L_PD_S_dxdy (annotation)",
        title="PD: Bounding box vs annotation distance"
    )

    scatter_with_fits(
        df["L_PD_midline"], df["L_PD_S_dxdy"],
        xlabel="L_PD_midline (mesh)",
        ylabel="L_PD_S_dxdy (annotation)",
        title="PD: Midline vs annotation distance"
    )


# ============================================================
# Main
# ============================================================

def main():
    csv_file = os.path.join(
        scalar_path,
        "scalarGrowthData_meshBased.csv"
    )

    df = pd.read_csv(csv_file)
    print(df.head())
    print(f"Loaded {len(df)} samples")

    analyze_axes(df)
    analyze_variability(df)

    annotation_file = os.path.join(
        scalar_path,
        "shivani_annotation.xlsx"
    )

    if os.path.exists(annotation_file):
        df_anno = pd.read_excel(annotation_file)
        print("\nLoaded shivani_annotation.xlsx")
        print(df_anno.head())
    else:
        print("\nWARNING: shivani_annotation.xlsx not found")
        df_anno = None

    # -------------------------
    # Prepare annotation key
    # -------------------------
    df_anno = df_anno.copy()
    df_anno["Mask Folder"] = df_anno["name"].str.replace("_FlatFin$", "", regex=True)

    # -------------------------
    # Inner join
    # -------------------------
    df_joined = df.merge(
        df_anno,
        on="Mask Folder",
        how="inner"
    )

    print("\nJoined DataFrame (inner join on Mask Folder):")
    print(df_joined.head())

    print(f"\nJoined samples: {len(df_joined)} / {len(df)} and {len(df_anno)}")

    # missing_in_df = sorted(
    #     set(df_anno["Mask Folder"]) - set(df["Mask Folder"])
    # )

    # print("\nAnnotation Mask Folders not found in df (showing up to 5):")
    # print(missing_in_df[:5])

    analyze_L_PD_S(df_joined)

if __name__ == "__main__":
    main()
