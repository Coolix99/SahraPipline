import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import arviz as az
import stan
import asyncio
from utilsScalar import getData


def explorative_plotting(df: pd.DataFrame):
    sns.scatterplot(data=df, x="time in hpf", y="Surface Area", hue="condition")
    plt.title("Surface Area over Time by Condition")
    plt.xlabel("Time (hpf)")
    plt.ylabel("Surface Area")
    plt.tight_layout()
    plt.show()


from scipy.integrate import solve_ivp


def ode_system(t, y, alpha, beta_, A_end, A_cut):
    A, g = y
    dA_dt = g * A
    if A < A_cut:
        dg_dt = -alpha * (g - beta_ * (A_end - A_cut) / A_end)
    else:
        dg_dt = -alpha * (g - beta_ * (A_end - A) / A_end)
    return [dA_dt, dg_dt]


def simulate_prior_predictive_ODE(t_vals, df: pd.DataFrame, n_samples=50):
    np.random.seed(42)

    plt.figure(figsize=(10, 6))

    for _ in range(n_samples):
        alpha_tilde = np.random.normal(-0.5, 0.5)
        beta_tilde = np.random.normal(0, 0.1)

        A_end_tilde = np.random.normal(1.0, 0.1)
        A_cut_tilde = np.random.normal(0, 1)

        A_0_tilde = np.random.normal(0.3, 0.15)
        g_0 = np.random.normal(0, 0.1)

        alpha = 10 ** alpha_tilde
        beta_ = (alpha / 4) * 10 ** beta_tilde
        A_end = 10 ** A_end_tilde
        A_cut = 2 + 4 * (1 / (1 + np.exp(-A_cut_tilde)))
        A_0 = 10 ** A_0_tilde

        sol = solve_ivp(
            lambda t, y: ode_system(t, y, alpha, beta_, A_end, A_cut),
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

import nest_asyncio


nest_asyncio.apply()  # Needed for environments like Jupyter/IPython

def run_async(coro):
    """Safely run an async coroutine even in an existing event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        if loop.is_running():
            return loop.create_task(coro)
        else:
            return loop.run_until_complete(coro)



def compile_and_fit_stan_model(model_code: str, data_dict: dict):
    posterior = stan.build(model_code, data=data_dict)
    fit = posterior.sample(num_chains=4, num_samples=1000, num_warmup=3000, show_console=True)
    return fit

def posterior_diagnostics(fit):
    az_data = az.from_pystan(posterior=fit)
    az.plot_trace(az_data)
    plt.tight_layout()
    plt.show()

    az.plot_pair(az_data, kind='kde', marginals=True)
    plt.tight_layout()
    plt.show()

    print(az.summary(az_data))

def prepare_stan_data(df: pd.DataFrame, cond_dev="Development", cond_reg="Regeneration") -> dict:
    df_dev = df[df["condition"] == cond_dev].copy()
    df_reg = df[df["condition"] == cond_reg].copy()

    # Sort Dev data by time
    df_dev = df_dev.sort_values("time in hpf")
    t_dev = df_dev["time in hpf"].to_numpy()
    A_dev = df_dev["Surface Area"].to_numpy() * 1e-4

    # Sort Reg data by time
    df_reg = df_reg.sort_values("time in hpf")
    t_reg = df_reg["time in hpf"].to_numpy()
    A_reg = df_reg["Surface Area"].to_numpy() * 1e-4

    t_ppc = np.linspace(48, 144, 100)

    return {
        "t_Dev": t_dev, "A_Dev": A_dev, "N_Dev": len(t_dev),
        "t_Reg": t_reg, "A_Reg": A_reg, "N_Reg": len(t_reg),
        "t_ppc": t_ppc, "N_ppc": len(t_ppc)
    }

def main():
    df = getData()
    df = df[["time in hpf", "Surface Area", "condition"]].dropna()

    explorative_plotting(df)
    t_range = np.linspace(df["time in hpf"].min(), df["time in hpf"].max(), 100)
    simulate_prior_predictive_ODE(t_range,df)
    

    # Load Stan model
    with open("3_fits/fit_setPoint.stan", "r") as f:
        model_code = f.read()

    data_dict = prepare_stan_data(df)


    fit = compile_and_fit_stan_model(model_code, data_dict)
    posterior_diagnostics(fit)


if __name__ == "__main__":
    main()
