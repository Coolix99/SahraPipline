import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

import matplotlib as mpl

mpl.rcParams.update({
    "svg.fonttype": "none",      # keep text editable in Illustrator
    "pdf.fonttype": 42,
    "text.usetex": False,
    "axes.unicode_minus": False
})

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

def getData():
    df_file_path = os.path.join(scalar_path,'scalarGrowthData_meshBased.csv')

    # Load the DataFrame from the HDF5 file
    df = pd.read_csv(df_file_path,sep=',')

    # Calculate new columns:
    # L AP * L PD
    df['L AP * L PD'] = df['L_AP_40line'] * df['L_PD_midline']

    # V / A (Volume / Surface Area)
    df['V / A'] = df['Volume'] / df['Surface Area']

    # Int_dA_d / A (Integrated Thickness / Surface Area)
    df['Int_dA_d / A'] = df['Integrated Thickness'] / df['Surface Area']

    df['log L AP'] = np.log(df['L_AP_40line'])
    df['log L PD'] = np.log(df['L_PD_midline'])

    return df


from powersmooth.powersmooth import powersmooth_general, upsample_with_mask

def fit_and_sample_derivatives(df: pd.DataFrame, column: str, N_samples: int = 200) -> dict:
    results = {}
    grouped = df.groupby('condition')

    for condition, group in grouped:

        if condition in {'4850cut', '7230cut'}:
            continue

        stats = group.groupby('time in hpf')[column].agg(['mean','std']).reset_index()

        x = stats['time in hpf'].values
        y_mean = stats['mean'].values
        y_std = stats['std'].values

        condition_fits = {
            'fitted_values': [],
            'derivative': [],
            'relative_derivative': [],
        }

        for _ in range(N_samples):

            y_sample = np.random.normal(y_mean, y_std/2) # type: ignore

            x_up, y_up, mask_up = upsample_with_mask(x, y_sample, dx=1) # type: ignore

            y_smooth = powersmooth_general(
                x_up,
                y_up,
                weights={2:7e2, 3:1e3},
                mask=mask_up
            )

            D1 = np.gradient(y_smooth, x_up)
            rel_D1 = D1 / y_smooth

            condition_fits['fitted_values'].append(y_smooth)
            condition_fits['derivative'].append(D1)
            condition_fits['relative_derivative'].append(rel_D1)

        condition_fits['time'] = x_up # type: ignore
        results[condition] = condition_fits

    return results

def fit_with_finite_differences(df: pd.DataFrame, column: str, N_samples: int = 500):

    results = {}
    grouped = df.groupby('condition')

    for condition, group in grouped:

        if condition in {'4850cut', '7230cut'}:
            continue

        stats = group.groupby('time in hpf')[column].agg(['mean','std']).reset_index()

        t = stats['time in hpf'].values
        y_mean = stats['mean'].values
        y_std = stats['std'].values

        t_dense = np.linspace(t.min(), t.max(), 400) # type: ignore

        fits = []
        derivatives = []
        rel_derivatives = []

        for _ in range(N_samples):

            y_sample = np.random.normal(y_mean, y_std/2) # type: ignore

            y_interp = np.interp(t_dense, t, y_sample) # type: ignore

            d = np.gradient(y_interp, t_dense)
            rel = d / y_interp

            fits.append(y_interp)
            derivatives.append(d)
            rel_derivatives.append(rel)

        results[condition] = {
            "time": t_dense,
            "fitted_values": fits,
            "derivative": derivatives,
            "relative_derivative": rel_derivatives
        }

    return results


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.interpolate import PchipInterpolator
def fit_with_monotonic_pchip(df: pd.DataFrame, column: str, N_samples: int = 500):
    results = {}
    grouped = df.groupby('condition')

    for condition, group in grouped:
        if condition in {'4850cut', '7230cut'}:
            continue

        stats = group.groupby('time in hpf')[column].agg(['mean', 'std']).reset_index()

        x = stats['time in hpf'].values.astype(float)
        y_mean = stats['mean'].values.astype(float)
        y_std = stats['std'].values.astype(float)

        x_dense = np.linspace(x.min(), x.max(), 400) # type: ignore

        fitted_values = []
        derivatives = []
        relative_derivatives = []

        for _ in range(N_samples):
            # sample noisy observations
            y_sample = np.random.normal(y_mean, y_std / 2) # type: ignore

            # enforce monotonicity on sampled support points
            # this is the key step: once the support points are monotone,
            # PCHIP preserves monotonicity between them
            y_sample_mono = np.maximum.accumulate(y_sample)

            interpolator = PchipInterpolator(x, y_sample_mono)

            y_dense = interpolator(x_dense)
            d_dense = interpolator.derivative()(x_dense)
            rel_d_dense = d_dense / y_dense

            fitted_values.append(y_dense)
            derivatives.append(d_dense)
            relative_derivatives.append(rel_d_dense)

        results[condition] = {
            'time': x_dense,
            'fitted_values': fitted_values,
            'derivative': derivatives,
            'relative_derivative': relative_derivatives
        }

    return results

from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid

def softplus(z):
    return np.logaddexp(0.0, z)

def inv_softplus(y):
    y = np.maximum(y, 1e-12)
    return np.log(np.expm1(y))

def build_rbf_basis(x, n_basis=8, width=None):
    x = np.asarray(x, dtype=float)
    centers = np.linspace(x.min(), x.max(), n_basis)

    if width is None:
        if n_basis > 1:
            spacing = centers[1] - centers[0]
            width = 1.5 * spacing
        else:
            width = x.ptp() if x.ptp() > 0 else 1.0

    Phi = np.exp(-0.5 * ((x[:, None] - centers[None, :]) / width) ** 2)

    # normalize rows so constant weights -> constant latent function
    Phi_sum = Phi.sum(axis=1, keepdims=True)
    Phi_sum[Phi_sum == 0] = 1.0
    Phi = Phi / Phi_sum

    return Phi, centers, width

def fit_monotone_softplus_single(
    x,
    y,
    y_std,
    x_dense,
    n_basis=8,
    smooth_penalty=1e-2
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    x_dense = np.asarray(x_dense, dtype=float)

    # avoid division issues
    y_std = np.where(np.isfinite(y_std) & (y_std > 0), y_std, np.nanmedian(y_std))
    if not np.isfinite(y_std).all():
        y_std = np.where(np.isfinite(y_std), y_std, 1.0)

    Phi_x, centers, width = build_rbf_basis(x, n_basis=n_basis)
    Phi_dense, _, _ = build_rbf_basis(x_dense, n_basis=n_basis, width=width)

    # initialize from average positive slope
    avg_slope = max((y[-1] - y[0]) / max(x[-1] - x[0], 1e-8), 1e-6)
    z0 = inv_softplus(avg_slope)

    p0 = np.zeros(n_basis + 1)
    p0[0] = max(y[0], 1e-8)     # y0
    p0[1:] = z0                  # latent derivative roughly constant initially

    def predict(params, x_eval_basis):
        y0 = params[0]
        w = params[1:]

        latent = x_eval_basis @ w
        dydt = softplus(latent)

        # integrate derivative to recover y
        y_eval = y0 + cumulative_trapezoid(dydt, x_dense if x_eval_basis is Phi_dense else x, initial=0.0)
        return y_eval, dydt

    def objective(params):
        y0 = params[0]
        w = params[1:]

        latent_x = Phi_x @ w
        dydt_x = softplus(latent_x)
        y_hat = y0 + cumulative_trapezoid(dydt_x, x, initial=0.0)

        resid = (y_hat - y) / y_std
        data_term = np.sum(resid ** 2)

        # smoothness penalty on latent weights
        if len(w) >= 3:
            d2w = np.diff(w, n=2)
            smooth_term = smooth_penalty * np.sum(d2w ** 2)
        else:
            smooth_term = 0.0

        # optional weak penalty against absurdly tiny y0
        y0_penalty = 1e-6 * (min(0.0, y0) ** 2)

        return data_term + smooth_term + y0_penalty

    res = minimize(
        objective,
        p0,
        method="L-BFGS-B",
        bounds=[(1e-10, None)] + [(None, None)] * n_basis
    )

    params_opt = res.x
    y0 = params_opt[0]
    w = params_opt[1:]

    latent_dense = Phi_dense @ w
    dydt_dense = softplus(latent_dense)
    y_dense = y0 + cumulative_trapezoid(dydt_dense, x_dense, initial=0.0)
    rel_dydt_dense = dydt_dense / y_dense

    return y_dense, dydt_dense, rel_dydt_dense

def fit_with_monotone_softplus(
    df: pd.DataFrame,
    column: str,
    N_samples: int = 500,
    n_basis: int = 8,
    smooth_penalty: float = 1e-2
):
    results = {}
    grouped = df.groupby('condition')

    for condition, group in grouped:
        if condition in {'4850cut', '7230cut'}:
            continue

        stats = group.groupby('time in hpf')[column].agg(['mean', 'std']).reset_index()

        x = stats['time in hpf'].values.astype(float)
        y_mean = stats['mean'].values.astype(float)
        y_std = stats['std'].values.astype(float)

        # replace missing/zero std values safely
        finite_pos_std = y_std[np.isfinite(y_std) & (y_std > 0)] # type: ignore
        fallback_std = np.median(finite_pos_std) if len(finite_pos_std) > 0 else 1.0
        y_std = np.where(np.isfinite(y_std) & (y_std > 0), y_std, fallback_std) # type: ignore

        x_dense = np.linspace(x.min(), x.max(), 400) # type: ignore

        fitted_values = []
        derivatives = []
        relative_derivatives = []

        for _ in range(N_samples):
            y_sample = np.random.normal(y_mean, y_std / 2.0) # type: ignore

            # keep values positive if needed
            if np.any(y_sample <= 0):
                y_sample = np.maximum(y_sample, 1e-8)

            y_fit, d_fit, rel_d_fit = fit_monotone_softplus_single(
                x=x,
                y=y_sample,
                y_std=y_std,
                x_dense=x_dense,
                n_basis=n_basis,
                smooth_penalty=smooth_penalty
            )

            fitted_values.append(y_fit)
            derivatives.append(d_fit)
            relative_derivatives.append(rel_d_fit)

        results[condition] = {
            'time': x_dense,
            'fitted_values': fitted_values,
            'derivative': derivatives,
            'relative_derivative': relative_derivatives
        }

    return results

def clean_limits(vmin, vmax, step):
    vmin = step * np.floor(vmin / step)
    vmax = step * np.ceil(vmax / step)
    return vmin, vmax
def plot_fit_with_uncertainty(
    fit_results_list, colors, labels, title, xlabel, ylabel,
    ymin=0, ymax=None, xmax=None, xmin=None, dy=None, dx=None,
    df=None, column=None
):
    """
    Plot fit results with uncertainty bands using Matplotlib.

    Parameters:
    -----------
    fit_results_list : list of dict
        List of fit result dictionaries. Each should have 't_values' and 'fits'.
    colors : list of str
        List of colors corresponding to each category.
    labels : list of str
        List of labels for the legend.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    """
    fig = plt.figure(figsize=(6, 4))

    for fit_results, color, label in zip(fit_results_list, colors, labels):
        t_values = fit_results['t_values']
        fits = fit_results['fits']

        fit_lower2sig, fit_lowersig, fit_mean, fit_uppersig, fit_upper2sig = np.percentile(
            fits, [2.3, 15.9, 50, 84.1, 97.7], axis=0
        )

        plt.plot(t_values, fit_mean, color=color, linewidth=2, label=label)

        # plot raw data
        if df is not None and column is not None:

            cond = label
            data = df[df["condition"] == cond]

            stats = data.groupby("time in hpf")[column].agg(["mean","std"]).reset_index()

            plt.errorbar(
                stats["time in hpf"],
                stats["mean"],
                yerr=stats["std"],
                fmt="o",
                color=color,
                markersize=5,
                capsize=3,
                linestyle="none"
            )
        plt.fill_between(t_values, fit_lowersig, fit_uppersig, color=color, alpha=0.2)
        #plt.fill_between(t_values, fit_lower2sig, fit_upper2sig, color=color, alpha=0.1)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=22)

    if ymin is None:
        ymin = plt.ylim()[0]
    if ymax is None:
        ymax = plt.ylim()[1]
    if xmin is None:
        xmin = plt.xlim()[0]
    if xmax is None:
        xmax = plt.xlim()[1]
    if dx is None:
        dx = (xmax - xmin) / 10
    if dy is None:
        dy = (ymax - ymin) / 10
    
    ymin, ymax = clean_limits(ymin, ymax, dy)
    xmin, xmax = clean_limits(xmin, xmax, dx)
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.xticks(np.arange(xmin, xmax, dx), fontsize=16)
    plt.yticks(np.arange(ymin, ymax, dy), fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()

COLORS = {
    "Development": "#2278b5",
    "Regeneration_30_48": "#f57f20",
    "Regeneration_50_48": "#017c91",
    "Regeneration_30_72": "#b29dcb"
}

def fit_with_gp(df: pd.DataFrame, column: str, N_samples: int = 500) -> dict:
    results = {}
    grouped = df.groupby('condition')

    for condition, group in grouped:
        if condition in {'4850cut', '7230cut'}:
            continue

        stats = group.groupby('time in hpf')[column].agg(['mean', 'std']).reset_index()

        X = stats['time in hpf'].values.reshape(-1, 1) # type: ignore
        y = stats['mean'].values
        y_std = stats['std'].values
        print(np.mean(y)) # type: ignore
        print(np.mean(y_std)) # type: ignore
        # Define kernel
        # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=20.0, length_scale_bounds=(1, 200))
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=50.0, length_scale_bounds="fixed")
        y_scale = np.mean(y) # type: ignore
        y = y / y_scale
        y_std = y_std / y_scale

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=y_std**2+ 1e-8,   # observational noise
            normalize_y=True,
            n_restarts_optimizer=5
        )

        gp.fit(X, y)

        # Extract fitted kernel
        kernel_opt = gp.kernel_
        # Extract RBF length scale
        length_scale = kernel_opt.k2.length_scale # type: ignore
        print(f"[{condition}] GP length scale (correlation time): {length_scale:.2f} hpf")

        # Dense time grid
        X_dense = np.linspace(X.min(), X.max(), 400).reshape(-1, 1) # type: ignore

        # Draw posterior samples
        y_samples = gp.sample_y(X_dense, n_samples=N_samples)

        fitted_values = []
        derivatives = []
        relative_derivatives = []

        for i in range(N_samples):
            y_s = y_samples[:, i]

            D1 = np.gradient(y_s, X_dense.flatten())
            rel_D1 = D1 / y_s

            y_s = y_s * y_scale
            D1 = D1 * y_scale

            fitted_values.append(y_s)
            derivatives.append(D1)
            relative_derivatives.append(rel_D1)

        results[condition] = {
            'time': X_dense.flatten(),
            'fitted_values': fitted_values,
            'derivative': derivatives,
            'relative_derivative': relative_derivatives
        }

    return results

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu


def p_to_stars(p):
    if p <= 0.001:
        return "***"
    elif p <= 0.01:
        return "**"
    elif p <= 0.05:
        return "*"
    else:
        return ""


def plot_volume_data_only(
    df: pd.DataFrame,
    column: str = "Volume",
    conditions=("Development", "Regeneration"),
    colors=None
):
    if colors is None:
        colors = {
            "Development": "#2278b5",
            "Regeneration": "#f57f20"
        }

    plt.figure(figsize=(7, 5))

    all_ns = {cond: [] for cond in conditions}

    # ----------- PLOT DATA -----------
    for condition in conditions:
        group = df[df["condition"] == condition]

        stats = (
            group.groupby("time in hpf")[column]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        t = stats["time in hpf"].values
        mean = stats["mean"].values / 1e6
        std = stats["std"].values / 1e6
        counts = stats["count"].values

        # print n per time point
        for ti, ni in zip(t, counts):
            all_ns[condition].append(ni)
            print(f"{condition} | t={ti}: n={ni}")

        # shaded std
        plt.fill_between(
            t,
            mean - std,
            mean + std,
            color=colors[condition],
            alpha=0.2,
            linewidth=0
        )

        # line
        plt.plot(
            t,
            mean,
            color=colors[condition],
            linewidth=2,
            zorder=2
        )

        # points
        plt.scatter(
            t,
            mean,
            color=colors[condition],
            edgecolor="white",
            linewidth=1.0,
            s=100,
            zorder=3,
            label=condition
        )

    # ----------- STATISTICS -----------
    cond1, cond2 = conditions

    group1 = df[df["condition"] == cond1]
    group2 = df[df["condition"] == cond2]

    common_times = sorted(
        set(group1["time in hpf"]).intersection(group2["time in hpf"])
    )

    for t_val in common_times:
        data1 = group1[group1["time in hpf"] == t_val][column].values
        data2 = group2[group2["time in hpf"] == t_val][column].values

        if len(data1) > 0 and len(data2) > 0:
            stat, p = mannwhitneyu(data1, data2, alternative="two-sided")
            stars = p_to_stars(p)

            if stars != "":
                m1 = np.mean(data1) / 1e6
                m2 = np.mean(data2) / 1e6
                y = max(m1, m2)

                y_offset = 0.15

                for i, _ in enumerate(stars):
                    plt.text(
                        t_val,
                        y + y_offset + i * 0.12,
                        "*",
                        ha="center",
                        va="bottom",
                        fontsize=18,
                        color="black"
                    )

    # ----------- AXES -----------
    plt.xlabel("Developmental time [hpf]", fontsize=20)
    plt.ylabel(r"Fin Volume$\;[10^6 \, \mu m^3]$", fontsize=20)

    xticks = [48, 60, 72, 84, 96, 108, 120, 132, 144]
    plt.xticks(xticks, rotation=45, fontsize=16)

    yticks = np.arange(0.0, 4.5, 0.5)
    plt.yticks(yticks, fontsize=16)

    plt.ylim(0, 4.0)

    # no grid
    plt.grid(False)

    # ----------- PRINT TOTAL N -----------
    print("\n--- Total sample sizes ---")
    for cond in conditions:
        print(f"{cond}: total n = {sum(all_ns[cond])}")

    plt.tight_layout()
    plt.show()

def plot_gp_rate(
    fit_results_list,
    colors,
    labels,
    ylabel,
    scale_y=1.0,
    ylim=None,
    yticks=None
):
    plt.figure(figsize=(7, 5))

    for fit_res, color, label in zip(fit_results_list, colors, labels):
        t = fit_res["t_values"]

        fits = np.array(fit_res["fits"]) / scale_y  # shape: (samples, time)

        mean = np.mean(fits, axis=0)
        std = np.std(fits, axis=0)

        # --- shaded uncertainty ---
        plt.fill_between(
            t,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
            linewidth=0
        )

        # --- mean curve ---
        plt.plot(
            t,
            mean,
            color=color,
            linewidth=2.5,
            label=label
        )

    # ----------- AXES -----------
    plt.xlabel("Developmental time [hpf]", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    xticks = [48, 60, 72, 84, 96, 108, 120, 132, 144]
    plt.xticks(xticks, rotation=45, fontsize=16)

    if yticks is not None:
        plt.yticks(yticks, fontsize=16)
    else:
        plt.yticks(fontsize=16)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(False)
    ax = plt.gca()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_bounds(yticks[0], yticks[-1]) # type: ignore
    xmin, xmax = ax.get_xlim()
    ax.spines["bottom"].set_bounds(xmin, xticks[-1])
    plt.subplots_adjust(
        left=0.15,
        right=1.0,
        bottom=0.2025,
        top=1.0
    )
    plt.show()

def main():

    df = getData()
    print(df.columns)

    plot_volume_data_only(
        df=df,
        column="Volume",
        conditions=("Development", "Regeneration"),
        colors={
            "Development": COLORS["Development"],
            "Regeneration": COLORS["Regeneration_30_48"]
        }
    )

    # interpolation / fitting methods
    methods = {
        # "MonotonicSoftplus": fit_with_monotone_softplus,
        # "FiniteDifference": fit_with_finite_differences,
        # "PowerSmooth": fit_and_sample_derivatives,
        "GaussianProcess": fit_with_gp,
        # "MonotonicPCHIP": fit_with_monotonic_pchip
    }

    # variables to analyze
    variables = {
        "Volume": {
            "ylabel_value": r"$V\;[\mu m^3]$",
            "ylabel_rate": r"$\dot{V}\;[\mu m^3/h]$",
            "ylabel_rel": r"$\dot{V}/V\;[1/h]$",
            "dy_rate": 10000
        },
        # "Surface Area": {
        #     "ylabel_value": r"$A\;[\mu m^2]$",
        #     "ylabel_rate": r"$\dot{A}\;[\mu m^2/h]$",
        #     "ylabel_rel": r"$\dot{A}/A\;[1/h]$",
        #     "dy_rate": 1000
        # }
    }

    for method_name, method_func in methods.items():
        print(f"\nRunning method: {method_name}")
        for variable, meta in variables.items():
            print(f"  Variable: {variable}")
            res = method_func(df, variable, N_samples=1000)

            dev = res['Development']
            reg = res['Regeneration']

            # # ---------- VALUE ----------
            # plot_fit_with_uncertainty(
            #     fit_results_list=[
            #         {"t_values": dev["time"], "fits": dev["fitted_values"]},
            #         {"t_values": reg["time"], "fits": reg["fitted_values"]}
            #     ],
            #     colors=[COLORS["Development"], COLORS["Regeneration_30_48"]],
            #     labels=["Development", "Regeneration"],
            #     title=f"{variable} ({method_name})",
            #     xlabel="t [hpf]",
            #     ylabel=meta["ylabel_value"],
            #     df=df,
            #     column=variable
            # )

            # # ---------- ABSOLUTE RATE ----------
            # plot_fit_with_uncertainty(
            #     fit_results_list=[
            #         {"t_values": dev["time"], "fits": dev["derivative"]},
            #         {"t_values": reg["time"], "fits": reg["derivative"]}
            #     ],
            #     colors=[COLORS["Development"], COLORS["Regeneration_30_48"]],
            #     labels=["Development", "Regeneration"],
            #     title=f"{variable} Growth Rate ({method_name})",
            #     xlabel="t [hpf]",
            #     ylabel=meta["ylabel_rate"],
            #     ymin=None, # type: ignore
            #     xmin=48,
            #     dx=12,
            #     dy=meta["dy_rate"]
            # )

            # # ---------- RELATIVE RATE ----------
            # plot_fit_with_uncertainty(
            #     fit_results_list=[
            #         {"t_values": dev["time"], "fits": dev["relative_derivative"]},
            #         {"t_values": reg["time"], "fits": reg["relative_derivative"]}
            #     ],
            #     colors=[COLORS["Development"], COLORS["Regeneration_30_48"]],
            #     labels=["Development", "Regeneration"],
            #     title=f"Relative {variable} Growth Rate ({method_name})",
            #     xlabel="t [hpf]",
            #     ylabel=meta["ylabel_rel"],
            #     ymin=None, # type: ignore
            #     xmin=48,
            #     dx=12,
            #     dy=0.02
            # )


            plot_gp_rate(
                fit_results_list=[
                    {"t_values": dev["time"], "fits": dev["derivative"]},
                    {"t_values": reg["time"], "fits": reg["derivative"]}
                ],
                colors=[COLORS["Development"], COLORS["Regeneration_30_48"]],
                labels=["Development", "Regeneration"],
                ylabel=r"$\dot{V}\;[10^4 \, \mu m^3/h]$",
                scale_y=1e4, 
                ylim=(0, 4.5),   # adjust based on your data
                yticks=np.arange(0, 4.5, 0.5)
            )

            plot_gp_rate(
                fit_results_list=[
                    {"t_values": dev["time"], "fits": dev["relative_derivative"]},
                    {"t_values": reg["time"], "fits": reg["relative_derivative"]}
                ],
                colors=[COLORS["Development"], COLORS["Regeneration_30_48"]],
                labels=["Development", "Regeneration"],
                ylabel=r"$\dot{V}/V\;[1/h]$",
                scale_y=1.0,
                ylim=(0, 0.043),   # adjust if needed
                yticks=np.arange(0, 0.041, 0.01)
            )

if __name__ == "__main__":
    main()
