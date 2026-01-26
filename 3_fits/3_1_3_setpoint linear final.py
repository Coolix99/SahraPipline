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
from utilsScalar import getData, scalar_path
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
    df = pd.read_csv('/home/max/Downloads/Fin_measurements.csv',sep=',')
    # select only the columns we need
    df = df[['time in hpf', 'condition', 'Surface Area']]
    return df


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
    az_data = az.from_cmdstanpy(posterior=fit, posterior_predictive=["A_Dev_ppc", "A_Reg_ppc"])
    samples = az_data
    f_ppc_dev = samples.posterior_predictive.A_Dev_ppc.stack({"sample": ("chain", "draw")}).transpose("sample", "A_Dev_ppc_dim_0")
    f_ppc_reg = samples.posterior_predictive.A_Reg_ppc.stack({"sample": ("chain", "draw")}).transpose("sample", "A_Reg_ppc_dim_0")


    output_file("corner_Dev_linear.html")
    save(corner(samples, parameters=["A_end", "A_0_Dev", "g_0_Dev", "alpha", "beta_"], xtick_label_orientation=np.pi/4))

    output_file("corner_Reg_linear.html")
    save(corner(samples, parameters=["A_end", "A_0_Reg", "g_0_Reg", "alpha", "beta_"], xtick_label_orientation=np.pi/4))

    output_file("ppc_Dev_linear.html")
    save(predictive_regression(
        f_ppc_dev,
        samples_x=np.array(data_dict["t_ppc"]),
        data=np.dstack((data_dict["t_Dev"], data_dict["A_Dev"])).squeeze(),
        x_axis_label='t',
        y_axis_label='A'))

    output_file("ppc_Reg_linear.html")
    save(predictive_regression(
        f_ppc_reg,
        samples_x=np.array(data_dict["t_ppc"]),
        data=np.dstack((data_dict["t_Reg"], data_dict["A_Reg"])).squeeze(),
        x_axis_label='t',
        y_axis_label='A'))

    # Save posterior samples to CSV
    posterior = samples.posterior
    scalar_vars = [
        var for var in posterior.data_vars
        if posterior[var].ndim == 2   # (chain, draw)
    ]
    results = {
        var: posterior[var].values.reshape(-1)
        for var in scalar_vars
    }
    results_df = pd.DataFrame(results)

    fit_results_path = os.path.join(scalar_path, "fit_results")
    os.makedirs(fit_results_path, exist_ok=True)
    results_df.to_csv(os.path.join(fit_results_path, "area_sampled_parameter_results_setPoint_linear.csv"), index=False)

    # sigma_samples = samples.posterior["sigma"].values.flatten()
    # sigma_mean = np.mean(sigma_samples)
    # sigma_std = np.std(sigma_samples)
    # print(f"Posterior sigma: mean = {sigma_mean:.4f}, std = {sigma_std:.4f}")

def prepare_stan_data(df: pd.DataFrame, cond_dev="Development", cond_reg="Regeneration") -> dict:
    df_dev = df[df["condition"] == cond_dev].copy()
    df_reg = df[df["condition"] == cond_reg].copy()

    # Sort Dev data by time
    df_dev = df_dev.sort_values("time in hpf")
    t_dev = df_dev["time in hpf"].to_numpy()
    A_dev = df_dev["Surface Area"].to_numpy() 

    # Sort Reg data by time
    df_reg = df_reg.sort_values("time in hpf")
    t_reg = df_reg["time in hpf"].to_numpy()
    A_reg = df_reg["Surface Area"].to_numpy() 

    t_ppc = np.linspace(48, 144, 100)

    return {
        "t_Dev": t_dev, "A_Dev": A_dev, "N_Dev": len(t_dev),
        "t_Reg": t_reg, "A_Reg": A_reg, "N_Reg": len(t_reg),
        "t_ppc": t_ppc, "N_ppc": len(t_ppc)
    }

def plot_fit_from_csv(
    df: pd.DataFrame,
    csv_path: str,
    cond_dev="Development",
    cond_reg="Regeneration",
    n_ppc_draws=None,   # optionally subsample posterior
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.integrate import solve_ivp

    # -------------------------
    # Style (gemdata-like)
    # -------------------------
    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    palette = {
        cond_dev: sns.color_palette("tab10")[0],
        cond_reg: sns.color_palette("tab10")[1],
    }

    # -------------------------
    # Load posterior
    # -------------------------
    post = pd.read_csv(csv_path)

    if n_ppc_draws is not None and n_ppc_draws < len(post):
        post = post.sample(n_ppc_draws, random_state=0)

    # -------------------------
    # ODE system
    # -------------------------
    def ode_system(t, y, alpha, beta_, A_end):
        A, g = y
        return [
            g * A,
            -alpha * (g - beta_ * (A_end - A) / A_end),
        ]

    # -------------------------
    # Time grid
    # -------------------------
    t_plot = np.linspace(
        df["time in hpf"].min(),
        df["time in hpf"].max(),
        200,
    )

    # -------------------------
    # Posterior predictive simulation
    # -------------------------
    def posterior_predictive(A0_key, g0_key, sigma_key, sigma_rel_key):
        ppc = []

        for _, row in post.iterrows():
            sol = solve_ivp(
                ode_system,
                [t_plot[0], t_plot[-1]],
                [row[A0_key], row[g0_key]],
                t_eval=t_plot,
                args=(row["alpha"], row["beta_"], row["A_end"]),
                method="RK45",
            )

            Ahat = sol.y[0]

            sigma = row[sigma_key] + row[sigma_rel_key] * Ahat
            A_ppc = np.random.normal(Ahat, sigma)

            ppc.append(A_ppc)

        ppc = np.asarray(ppc)
        return ppc.mean(axis=0), ppc.std(axis=0)

    # -------------------------
    # Dev & Reg PPC
    # -------------------------
    mean_dev, std_dev = posterior_predictive(
        "A_0_Dev", "g_0_Dev", "sigma_Dev", "sigma_rel_Dev"
    )
    mean_reg, std_reg = posterior_predictive(
        "A_0_Reg", "g_0_Reg", "sigma_Reg", "sigma_rel_Reg"
    )

    # -------------------------
    # Shared y-limit
    # -------------------------
    ymax = max(
        (mean_dev + std_dev).max(),
        (mean_reg + std_reg).max(),
        df["Surface Area"].max(),
    ) * 1.05

    # -------------------------
    # Single-panel plot
    # -------------------------
    def single_panel(cond, mean, std):
        plt.figure(figsize=(3.4, 2.4))
        sub = df[df["condition"] == cond]

        plt.scatter(
            sub["time in hpf"],
            sub["Surface Area"],
            color=palette[cond],
            alpha=0.6,
            s=15,
        )

        plt.plot(
            t_plot,
            mean,
            color=palette[cond],
            linewidth=2,
        )

        plt.fill_between(
            t_plot,
            mean - std,
            mean + std,
            color=palette[cond],
            alpha=0.25,
        )

        plt.xlabel("Developmental time [hpf]")
        plt.ylabel(r"Surface area $(100\,\mu\mathrm{m})^2$")
        plt.xticks(np.arange(48, 150, 12))
        plt.ylim(0, ymax)

        sns.despine()
        plt.tight_layout()
        plt.show()

    single_panel(cond_dev, mean_dev, std_dev)
    single_panel(cond_reg, mean_reg, std_reg)

    # -------------------------
    # Parameter summaries
    # -------------------------
    def summarize(vars_):
        return pd.DataFrame({
            "mean": post[vars_].mean(),
            "std": post[vars_].std(),
        })

    summary_dev = summarize(
        ["A_end", "alpha", "beta_", "A_0_Dev", "g_0_Dev", "sigma_Dev", "sigma_rel_Dev"]
    )
    summary_reg = summarize(
        ["A_end", "alpha", "beta_", "A_0_Reg", "g_0_Reg", "sigma_Reg", "sigma_rel_Reg"]
    )

    print("\n=== Development (posterior mean ± std) ===")
    print(summary_dev)

    print("\n=== Regeneration (posterior mean ± std) ===")
    print(summary_reg)

    return summary_dev, summary_reg


def main():
    df = getData()
    df['Surface Area'] = df['Surface Area'] / 10000


    # explorative_plotting(df)
    # t_range = np.linspace(df["time in hpf"].min(), df["time in hpf"].max(), 100)
    # simulate_prior_predictive_ODE(t_range,df)
    
    # data_dict = prepare_stan_data(df)

    # model_path = "3_fits/fit_setPoint_linear_final.stan"
    # fit = compile_and_fit_stan_model(model_path, data_dict)
    # posterior_diagnostics(fit,data_dict)

    plot_fit_from_csv(
        df,
        csv_path=os.path.join(
            scalar_path,
            "fit_results",
            "area_sampled_parameter_results_setPoint_linear.csv",
        ),
    )


if __name__ == "__main__":
    main()
