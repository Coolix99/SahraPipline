import os
import matplotlib.pyplot as plt
import pandas as pd
from requests import post
from requests import post
import seaborn as sns
import numpy as np
import arviz as az
from cmdstanpy import CmdStanModel
from bebi103.viz import corner, predictive_regression
from bokeh.io import output_file, save
from scipy.integrate import solve_ivp
from utilsScalar import scalar_path
from bokeh.io import output_file, save

from scipy.stats import mannwhitneyu

def explorative_plotting(
    df: pd.DataFrame,
    conditions_to_plot=None,   # NEW
):
    # ----------- colors -----------
    colors = {
        "Development": "#2077b5",
        "Regeneration": "#f57e1f",
        "4850cut": "#5b3a2a",
        "7230cut": "#d8b07a",
        "smoc_dev": "#7b5ea7",
        "smoc_reg": "#df3066",
    }

    if conditions_to_plot is None:
        conditions_to_plot = df["condition"].unique()

    df = df[df["condition"].isin(conditions_to_plot)]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    all_ns = {cond: [] for cond in conditions_to_plot}

    # ----------- PLOT DATA -----------
    for cond in conditions_to_plot:
        sub = df[df["condition"] == cond]

        stats = (
            sub.groupby("time in hpf")["Surface Area"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        t = stats["time in hpf"].values
        mean = stats["mean"].values
        std = stats["std"].values
        counts = stats["count"].values

        # print n
        for ti, ni in zip(t, counts):
            all_ns[cond].append(ni)
            print(f"{cond} | t={ti}: n={ni}")

        # shaded std
        ax.fill_between(
            t,
            mean - std,
            mean + std,
            color=colors.get(cond, "gray"),
            alpha=0.2,
            linewidth=0
        )

        # line
        ax.plot(
            t,
            mean,
            color=colors.get(cond, "gray"),
            linewidth=2,
            zorder=2
        )

        # points
        ax.scatter(
            t,
            mean,
            color=colors.get(cond, "gray"),
            edgecolor="white",
            linewidth=1.2,
            s=120,
            zorder=3,
            label=cond
        )

    # ----------- SIGNIFICANCE (only if exactly 2 conditions) -----------
    if len(conditions_to_plot) == 2:
        cond1, cond2 = conditions_to_plot
        df1 = df[df["condition"] == cond1]
        df2 = df[df["condition"] == cond2]

        common_times = sorted(
            set(df1["time in hpf"]).intersection(df2["time in hpf"])
        )

        def p_to_stars(p):
            if p <= 0.001:
                return "***"
            elif p <= 0.01:
                return "**"
            elif p <= 0.05:
                return "*"
            else:
                return ""

        for t_val in common_times:
            d1 = df1[df1["time in hpf"] == t_val]["Surface Area"].values
            d2 = df2[df2["time in hpf"] == t_val]["Surface Area"].values

            if len(d1) > 0 and len(d2) > 0:
                _, p = mannwhitneyu(d1, d2, alternative="two-sided")
                stars = p_to_stars(p)

                if stars:
                    m1 = np.mean(d1)
                    m2 = np.mean(d2)
                    y = max(m1, m2)

                    for i, _ in enumerate(stars):
                        ax.text(
                            t_val,
                            y + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]) + i * 0.04,
                            "*",
                            ha="center",
                            va="bottom",
                            fontsize=18,
                            color="black"
                        )

    # ----------- AXES -----------
    ax.set_xlabel("Developmental time [hpf]", fontsize=20)
    ax.set_ylabel("Surface Area", fontsize=20)

    xticks = [48, 60, 72, 84, 96, 108, 120, 132, 144]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45, fontsize=16) # type: ignore

    ax.tick_params(axis='y', labelsize=16)

    # --- styling ---
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_bounds(xticks[0], xticks[-1])

    plt.subplots_adjust(
        left=0.15,
        right=0.97,
        bottom=0.2025,
        top=0.979
    )

    # ----------- PRINT TOTAL N -----------
    print("\n--- Total sample sizes ---")
    for cond in conditions_to_plot:
        print(f"{cond}: total n = {sum(all_ns[cond])}")

    ax.legend(frameon=False, fontsize=14)

    #plt.show()
    return fig


def getData():
    # -------------------------
    # Base data
    # -------------------------
    df = pd.read_csv(
        os.path.join(scalar_path, "Fin_measurements.csv"), sep=","
    )[["time in hpf", "condition", "Surface Area"]]

    # -------------------------
    # Mesh-based cut data
    # -------------------------
    df2 = pd.read_csv(
        os.path.join(scalar_path, "scalarGrowthData_meshBased.csv"), sep=","
    )[["time in hpf", "condition", "Surface Area"]]

    cut_conditions = ["4850cut", "7230cut"]
    df2 = df2[df2["condition"].isin(cut_conditions)]

    # -------------------------
    # SMOC development data
    # -------------------------
    df_smoc_dev = pd.read_excel(
        os.path.join(scalar_path, "Smoc1_Smoc2_dev.xlsx")
    )[["Time in hpf", "surface_area"]]

    df_smoc_dev = df_smoc_dev.rename(columns={
        "Time in hpf": "time in hpf",
        "surface_area": "Surface Area",
    })

    df_smoc_dev["condition"] = "smoc_dev"

    # -------------------------
    # SMOC regeneration data
    # -------------------------
    df_smoc_reg = pd.read_excel(
        os.path.join(scalar_path, "Smoc1_Smoc2_reg_final.xlsx")
    )[["time in hpf", "Surface Area"]]

    df_smoc_reg["condition"] = "smoc_reg"

    # -------------------------
    # Combine all datasets
    # -------------------------
    df_all = pd.concat(
        [df, df2, df_smoc_dev, df_smoc_reg],
        ignore_index=True
    )

    return df_all

def ode_system(t, y, alpha, beta_, A_end):
    A, g = y
    dA_dt = g * A
    dg_dt = -alpha * (g - beta_ * (A_end - A) / A_end)
    return [dA_dt, dg_dt]

def simulate_prior_predictive_ODE(t_vals, df: pd.DataFrame, n_samples=50):
    np.random.seed(42)

    plt.figure(figsize=(10, 6))

    for _ in range(n_samples):
        alpha_tilde = np.random.normal(-0.5, 0.5)
        beta_tilde = np.random.normal(0, 0.1)

        A_end_tilde = np.random.normal(1.0, 0.1)

        A_0_tilde = np.random.normal(0.3, 0.15)
        g_0 = np.random.normal(0, 0.1)

        alpha = 10 ** alpha_tilde
        beta_ = (alpha / 4) * 10 ** beta_tilde
        A_end = 10 ** A_end_tilde
        A_0 = 10 ** A_0_tilde

        sol = solve_ivp(
            lambda t, y: ode_system(t, y, alpha, beta_, A_end),
            [t_vals[0], t_vals[-1]],
            [A_0, g_0],
            t_eval=t_vals,
            method="RK45",
            vectorized=False
        )

        plt.plot(sol.t, sol.y[0], color="gray", alpha=0.3)

    # Overlay real data
    sns.scatterplot(data=df, x="time in hpf", y="Surface Area", hue="condition", palette="tab10")

    plt.title("Prior Predictive Trajectories vs Observed Data")
    plt.xlabel("Time (hpf)")
    plt.ylabel("Surface Area A(t)")
    plt.tight_layout()
    plt.show()

def compile_and_fit_stan_model(model_path: str, data_dict: dict):
    model = CmdStanModel(stan_file=model_path)
    fit = model.sample(data=data_dict, chains=4, iter_warmup=3000, iter_sampling=1000, show_progress=True, show_console=False)
    return fit

def posterior_diagnostics(fit, data_dict):
    samples = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=[
            "A_Dev_ppc",
            "A_Reg_ppc",
            "A_4850cut_ppc",
            "A_7230cut_ppc",
        ],
    )

    # -------------------------------------------------
    # Helper: stack PPC samples
    # -------------------------------------------------
    def stack_ppc(var):
        return (
            samples.posterior_predictive[var] # type: ignore
            .stack(sample=("chain", "draw"))
            .transpose("sample", f"{var}_dim_0")
        )

    f_ppc = {
        "Dev": stack_ppc("A_Dev_ppc"),
        "Reg": stack_ppc("A_Reg_ppc"),
        "4850cut": stack_ppc("A_4850cut_ppc"),
        "7230cut": stack_ppc("A_7230cut_ppc"),
    }

    # -------------------------------------------------
    # Corner plots
    # -------------------------------------------------
    corner_specs = {
        "Dev": ["A_end", "A_0_Dev", "g_0_Dev", "alpha", "beta_"],
        "Reg": ["A_end", "A_0_Reg", "alpha", "beta_"],
        "4850cut": ["A_end", "A_0_4850cut", "alpha", "beta_"],
        "7230cut": ["A_end", "A_0_7230cut", "alpha", "beta_"],
    }

    for key, params in corner_specs.items():
        output_file(f"corner_{key}_linear.html")
        save(corner(samples, parameters=params, xtick_label_orientation=np.pi / 4)) # type: ignore

    # -------------------------------------------------
    # PPC plots
    # -------------------------------------------------
    ppc_specs = {
        "Dev": ("t_ppc_48", "t_Dev", "A_Dev"),
        "Reg": ("t_ppc_48", "t_Reg", "A_Reg"),
        "4850cut": ("t_ppc_48", "t_4850cut", "A_4850cut"),
        "7230cut": ("t_ppc_72", "t_7230cut", "A_7230cut"),
    }

    for key, (t_ppc_key, t_key, A_key) in ppc_specs.items():
        output_file(f"ppc_{key}_linear.html")
        save(
            predictive_regression(
                f_ppc[key],
                samples_x=np.asarray(data_dict[t_ppc_key]),
                data=np.dstack(
                    (data_dict[t_key], data_dict[A_key])
                ).squeeze(),
                x_axis_label="t",
                y_axis_label="A",
            )
        )

    # -------------------------------------------------
    # Save posterior samples to CSV
    # -------------------------------------------------
    posterior = samples.posterior # type: ignore
    scalar_vars = [
        v for v in posterior.data_vars if posterior[v].ndim == 2
    ]

    results_df = pd.DataFrame(
        {v: posterior[v].values.reshape(-1) for v in scalar_vars}
    )

    fit_results_path = os.path.join(scalar_path, "fit_results")
    os.makedirs(fit_results_path, exist_ok=True)

    results_df.to_csv(
        os.path.join(
            fit_results_path,
            "area_sampled_parameter_results_setPoint_linear_all.csv",
        ),
        index=False,
    )


def prepare_stan_data(
    df: pd.DataFrame,
    n_ppc_48=100,
    n_ppc_72=100,
) -> dict:
    """
    Prepare Stan data with explicit condition ↔ Stan-name mapping.
    """

    # Explicit mapping: Python condition → Stan suffix
    cond_map = {
        "Development": "Dev",
        "Regeneration": "Reg",
        "4850cut": "4850cut",
        "7230cut": "7230cut",
        "smoc_dev": "smoc_Dev",
        "smoc_reg": "smoc_Reg",
    }

    data = {}

    # -------------------------
    # Per-condition data
    # -------------------------
    for cond, stan_key in cond_map.items():
        sub = df[df["condition"] == cond].sort_values("time in hpf")

        t = sub["time in hpf"].to_numpy()
        A = sub["Surface Area"].to_numpy()

        data[f"t_{stan_key}"] = t
        data[f"A_{stan_key}"] = A
        data[f"N_{stan_key}"] = len(t)

    # -------------------------
    # PPC grids
    # -------------------------
    t_max = df["time in hpf"].max()

    # For Dev / Reg / 4850cut
    t_ppc_48 = np.linspace(48.0, t_max, n_ppc_48)

    # For 7230cut
    t_ppc_72 = np.linspace(72.0, t_max, n_ppc_72)

    data["t_ppc_48"] = t_ppc_48
    data["N_ppc_48"] = len(t_ppc_48)

    data["t_ppc_72"] = t_ppc_72
    data["N_ppc_72"] = len(t_ppc_72)

    return data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
def plot_fit_from_csv(
    df: pd.DataFrame,
    csv_path: str,
    conditions=None,
    n_ppc_draws=None,
):
    if conditions is None:
        conditions = df["condition"].unique()

    # ----------- COLORS (same as explorative) -----------
    colors = {
        "Development": "#2077b5",
        "Regeneration": "#f57e1f",
        "4850cut": "#5b3a2a",
        "7230cut": "#d8b07a",
        "smoc_dev": "#7b5ea7",
        "smoc_reg": "#df3066",
    }

    cond_map = {
        "Development": "Dev",
        "Regeneration": "Reg",
        "4850cut": "4850cut",
        "7230cut": "7230cut",
        "smoc_dev": "smoc_Dev",
        "smoc_reg": "smoc_Reg",
    }

    t0_map = {
        "Dev": 48.0,
        "Reg": 48.0,
        "4850cut": 48.0,
        "7230cut": 72.0,
        "smoc_Dev": 48.0,
        "smoc_Reg": 48.0,
    }

    post = pd.read_csv(csv_path)
    if n_ppc_draws is not None:
        post = post.sample(n_ppc_draws, random_state=0)

    def ode_system(t, y, alpha, beta_, A_end):
        A, g = y
        return [g * A, -alpha * (g - beta_ * (A_end - A) / A_end)]

    t_plot = np.linspace(df["time in hpf"].min(), df["time in hpf"].max(), 200)

    def posterior_predictive(stan_key):
        ppc = []

        for _, row in post.iterrows():
            g0 = row[f"g_0_{stan_key}"] if f"g_0_{stan_key}" in post.columns else 0.0
            
            # special A_end for smoc_dev
            if stan_key == "smoc_Dev":
                A_end_use = row["A_end_smocdev"]
            else:
                A_end_use = row["A_end"]

            sol = solve_ivp(
                ode_system,
                [t0_map[stan_key], t_plot[-1]],
                [row[f"A_0_{stan_key}"], g0],
                t_eval=t_plot[t_plot >= t0_map[stan_key]],
                args=(row["alpha"], row["beta_"], A_end_use),
            )

            Ahat = np.full_like(t_plot, np.nan, dtype=float)
            Ahat[t_plot >= t0_map[stan_key]] = sol.y[0]

            sigma = row[f"sigma_{stan_key}"] + row[f"sigma_rel_{stan_key}"] * Ahat
            ppc.append(np.random.normal(Ahat, sigma))

        ppc = np.asarray(ppc)
        return np.nanmean(ppc, axis=0), np.nanstd(ppc, axis=0)

    # ----------- PLOTTING -----------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for cond in conditions:
        stan_key = cond_map[cond]
        sub = df[df["condition"] == cond]

        mean, std = posterior_predictive(stan_key)

        # raw scatter
        ax.scatter(
            sub["time in hpf"],
            sub["Surface Area"],
            color=colors[cond],
            edgecolor="white",
            linewidth=1.0,
            s=80,
            alpha=0.6,
            zorder=3,
        )

        # fit mean
        ax.plot(
            t_plot,
            mean,
            color=colors[cond],
            linewidth=2.5,
            zorder=2,
        )

        # uncertainty
        ax.fill_between(
            t_plot,
            mean - std,
            mean + std,
            color=colors[cond],
            alpha=0.25,
            linewidth=0,
        )
    A_end_mean = post["A_end"].mean()
    A_end_smoc_mean = post["A_end_smocdev"].mean() if "A_end_smocdev" in post.columns else None
    # global A_end
    ax.axhline(
        A_end_mean,
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )

    # smoc-specific A_end
    if A_end_smoc_mean is not None and "smoc_dev" in conditions:
        ax.axhline(
            A_end_smoc_mean,
            color=colors["smoc_dev"],
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )

    alpha_mean = post["alpha"].mean()
    tau = 1 / alpha_mean   # characteristic time
   

    # ----------- AXES STYLE (MATCHED) -----------
    xticks = [48, 60, 72, 84, 96, 108, 120, 132, 144]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45, fontsize=16) # type: ignore

    # position (bottom right-ish)
    x0 = xticks[-1] - 40 # type: ignore
    y0 = 0.1 * ax.get_ylim()[1]

    # draw bar
    ax.plot(
        [x0, x0 + tau],
        [y0, y0],
        color="black",
        linewidth=3,
    )

    # label
    ax.text(
        x0 + tau / 2,
        y0 * 0.85,
        r"$\tau$",
        ha="center",
        va="top",
        fontsize=14,
    )

    ax.set_xlabel("Developmental time [hpf]", fontsize=20)
    ax.set_ylabel("Surface Area", fontsize=20)

    ax.tick_params(axis='y', labelsize=16)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xmin, xmax = ax.get_xlim()
    ax.spines["bottom"].set_bounds(xmin, xticks[-1]) # type: ignore

    plt.subplots_adjust(
        left=0.15,
        right=0.97,
        bottom=0.2025,
        top=0.979
    )

    #plt.show()
    

    # ----------- PARAMETER REPORT -----------
    print("\n=== Posterior summary ===")

    def summarize(vars_):
        return pd.DataFrame({
            "mean": post[vars_].mean(),
            "std": post[vars_].std(),
        })

    shared = ["A_end", "alpha", "beta_"]
    print("\nShared:")
    print(summarize(shared))

    if "A_end_smocdev" in post.columns:
        print("\nSMOC-specific:")
        print(summarize(["A_end_smocdev"]))

    for cond in conditions:
        stan_key = cond_map[cond]

        vars_cond = [
            f"A_0_{stan_key}",
            f"sigma_{stan_key}",
            f"sigma_rel_{stan_key}",
        ]

        g0_name = f"g_0_{stan_key}"
        if g0_name in post.columns:
            vars_cond.insert(1, g0_name)

        print(f"\n--- {cond} ---")
        print(summarize(vars_cond))
    return fig

def save_figure(fig, base_path):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    fig.savefig(base_path + ".png", dpi=300)
    fig.savefig(base_path + ".pdf")
    fig.savefig(base_path + ".svg")

    plt.close(fig)

def main():
    base_plot_path = os.path.join(scalar_path, "plots")
    condition_sets = {
        "all": None,
        # "dev_reg": ["Development", "Regeneration"],
        # "dev_4850": ["Development", "4850cut"],
        # "reg_4850": ["Regeneration", "4850cut"],
        # "dev_7230": ["Development", "7230cut"],
        # "dev_smocdev": ["Development", "smoc_dev"],
        # "reg_smocreg": ["Regeneration", "smoc_reg"],
        # "smocdev_smocreg": ["smoc_dev", "smoc_reg"],
        # "all_no_smoc": ["Development", "Regeneration", "4850cut", "7230cut"],
        # "dev"              : ["Development"],
        # "reg"              : ["Regeneration"],
        # "cut_4850"         : ["4850cut"],
        # "cut_7230"         : ["7230cut"],
        # "smoc_dev_only"    : ["smoc_dev"],
        # "smoc_reg_only"    : ["smoc_reg"],
    }
    df = getData()
    df['Surface Area'] = df['Surface Area'] / 10000
    print(df.head())
    # ----------- EXPLORATIVE -----------
    for name, conds in condition_sets.items():
        print(f"\n=== Explorative: {name} ===")

        fig = explorative_plotting(
            df,
            conditions_to_plot=conds,
        )

        save_figure(
            fig,
            os.path.join(base_plot_path, "explorative", name, "plot"),
        )
    t_range = np.linspace(df["time in hpf"].min(), df["time in hpf"].max(), 100)
    simulate_prior_predictive_ODE(t_range,df)
    
    data_dict = prepare_stan_data(df)

    # model_path = "3_fits/fit_setPoint_linear_all.stan"
    # fit = compile_and_fit_stan_model(model_path, data_dict)
    # posterior_diagnostics(fit,data_dict)
    csv_path = os.path.join(
        scalar_path,
        "fit_results",
        "area_sampled_parameter_results_setPoint_linear_all.csv",
    )
    # ----------- FIT PLOTS -----------
    for name, conds in condition_sets.items():
        print(f"\n=== Fit: {name} ===")

        fig = plot_fit_from_csv(
            df,
            csv_path=csv_path,
            conditions=conds,
        )

        save_figure(
            fig,
            os.path.join(base_plot_path, "fit", name, "plot"),
        )


if __name__ == "__main__":
    main()
