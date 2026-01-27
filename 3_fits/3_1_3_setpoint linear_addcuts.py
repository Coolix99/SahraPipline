import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import arviz as az
from cmdstanpy import CmdStanModel
from bebi103.viz import corner, predictive_regression
from bokeh.io import output_file, save
from scipy.integrate import solve_ivp
from utilsScalar import scalar_path
from bokeh.io import output_file, save

def explorative_plotting(df: pd.DataFrame):
    plt.figure(figsize=(8, 6))

    # Create a consistent color mapping per condition
    conditions = df["condition"].unique()
    palette = dict(zip(conditions, sns.color_palette("tab10", len(conditions))))

    # Scatter plot
    sns.scatterplot(
        data=df,
        x="time in hpf",
        y="Surface Area",
        hue="condition",
        palette=palette,
        alpha=0.5
    )

    # Compute mean and std
    stats = (
        df.groupby(["condition", "time in hpf"])["Surface Area"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Overlay mean ± std using the SAME colors
    for cond, sub in stats.groupby("condition"):
        color = palette[cond]

        plt.plot(
            sub["time in hpf"],
            sub["mean"],
            color=color,
            linewidth=2
        )

        plt.fill_between(
            sub["time in hpf"],
            sub["mean"] - sub["std"],
            sub["mean"] + sub["std"],
            color=color,
            alpha=0.25
        )

    plt.title("Surface Area over Time by Condition")
    plt.xlabel("Time (hpf)")
    plt.ylabel("Surface Area")
    plt.legend()
    plt.tight_layout()
    plt.show()

def getData():
    # Base data
    df = pd.read_csv(
        os.path.join(scalar_path, "Fin_measurements.csv"), sep=","
    )[["time in hpf", "condition", "Surface Area"]]

    # Additional mesh-based data
    df2 = pd.read_csv(
        os.path.join(scalar_path, "scalarGrowthData_meshBased.csv"), sep=","
    )[["time in hpf", "condition", "Surface Area"]]

    # Select only cut conditions from df2
    cut_conditions = ["4850cut", "7230cut"]
    df2 = df2[df2["condition"].isin(cut_conditions)]

    # Concatenate and return
    df_all = pd.concat([df, df2], ignore_index=True)

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
            samples.posterior_predictive[var]
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
        "Reg": ["A_end", "A_0_Reg", "g_0_Reg", "alpha", "beta_"],
        "4850cut": ["A_end", "A_0_4850cut", "g_0_4850cut", "alpha", "beta_"],
        "7230cut": ["A_end", "A_0_7230cut", "g_0_7230cut", "alpha", "beta_"],
    }

    for key, params in corner_specs.items():
        output_file(f"corner_{key}_linear.html")
        save(corner(samples, parameters=params, xtick_label_orientation=np.pi / 4))

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
    posterior = samples.posterior
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
            "area_sampled_parameter_results_setPoint_linear_addcut.csv",
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

def plot_fit_from_csv(
    df: pd.DataFrame,
    csv_path: str,
    conditions=("Development", "Regeneration", "4850cut", "7230cut"),
    n_ppc_draws=None,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.integrate import solve_ivp

    sns.set_style("white")

    
    cond_map = {
        "Development": "Dev",
        "Regeneration": "Reg",
        "4850cut": "4850cut",
        "7230cut": "7230cut",
    }

    t0_map = {
        "Dev": 48.0,
        "Reg": 48.0,
        "4850cut": 48.0,
        "7230cut": 72.0,
    }


    palette = dict(zip(conditions, sns.color_palette("tab10", len(conditions))))

    post = pd.read_csv(csv_path)
    if n_ppc_draws is not None:
        post = post.sample(n_ppc_draws, random_state=0)


    def ode_system(t, y, alpha, beta_, A_end):
        A, g = y
        return [
            g * A,
            -alpha * (g - beta_ * (A_end - A) / A_end),
        ]

    t_plot = np.linspace(
        df["time in hpf"].min(),
        df["time in hpf"].max(),
        200,
    )

    def posterior_predictive(stan_key):
        ppc = []

        t0 = t0_map[stan_key]   

        for _, row in post.iterrows():
            sol = solve_ivp(
                ode_system,
                [t0, t_plot[-1]],          
                [row[f"A_0_{stan_key}"], row[f"g_0_{stan_key}"]],
                t_eval=t_plot[t_plot >= t0], 
                args=(row["alpha"], row["beta_"], row["A_end"]),
            )

            # Fill full time axis with NaNs before t0 (for plotting)
            Ahat = np.full_like(t_plot, np.nan, dtype=float)
            Ahat[t_plot >= t0] = sol.y[0]

            sigma = (
                row[f"sigma_{stan_key}"]
                + row[f"sigma_rel_{stan_key}"] * Ahat
            )

            A_ppc = np.random.normal(Ahat, sigma)
            ppc.append(A_ppc)

        ppc = np.asarray(ppc)
        return np.nanmean(ppc, axis=0), np.nanstd(ppc, axis=0)



    ymax = df["Surface Area"].max() * 1.1

    for cond in conditions:
        stan_key = cond_map[cond]   # ← USE THE MAP
        mean, std = posterior_predictive(stan_key)
        sub = df[df["condition"] == cond]

        plt.figure(figsize=(3.4, 2.4))
        plt.scatter(
            sub["time in hpf"],
            sub["Surface Area"],
            color=palette[cond],
            alpha=0.6,
            s=15,
        )
        plt.plot(t_plot, mean, color=palette[cond], lw=2)
        plt.fill_between(t_plot, mean - std, mean + std, color=palette[cond], alpha=0.25)

        plt.title(cond)
        plt.xlabel("Developmental time [hpf]")
        plt.ylabel(r"Surface area $(100\,\mu\mathrm{m})^2$")
        plt.ylim(0, ymax)

        sns.despine()
        plt.tight_layout()
        plt.show()

    def summarize(vars_):
        return pd.DataFrame({
            "mean": post[vars_].mean(),
            "std": post[vars_].std(),
        })

    shared_params = ["A_end", "alpha", "beta_"]
    summary_shared = summarize(shared_params)
    print("\n=== Shared parameters (posterior mean ± std) ===")
    print(summary_shared)

    print("\n=== Condition-specific parameters ===")
    for cond in conditions:
        stan_key = cond_map[cond]

        vars_cond = [
            f"A_0_{stan_key}",
            f"g_0_{stan_key}",
            f"sigma_{stan_key}",
            f"sigma_rel_{stan_key}",
        ]

        summary_cond = summarize(vars_cond)

        print(f"\n--- {cond} ---")
        print(summary_cond)



def main():
    df = getData()
    df['Surface Area'] = df['Surface Area'] / 10000

    # explorative_plotting(df)
    # t_range = np.linspace(df["time in hpf"].min(), df["time in hpf"].max(), 100)
    # simulate_prior_predictive_ODE(t_range,df)
    
    # data_dict = prepare_stan_data(df)

    # model_path = "3_fits/fit_setPoint_linear_addcut.stan"
    # fit = compile_and_fit_stan_model(model_path, data_dict)
    # posterior_diagnostics(fit,data_dict)

    plot_fit_from_csv(
        df,
        csv_path=os.path.join(
            scalar_path,
            "fit_results",
            "area_sampled_parameter_results_setPoint_linear_addcut.csv",
        ),
    )


if __name__ == "__main__":
    main()
