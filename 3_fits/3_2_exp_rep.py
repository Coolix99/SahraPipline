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

# --- Explorative plotting ---
def explorative_plotting(df: pd.DataFrame):
    sns.scatterplot(data=df, x="time in hpf", y="Surface Area", hue="condition")
    plt.title("Surface Area over Time by Condition")
    plt.xlabel("Time (hpf)")
    plt.ylabel("Surface Area")
    plt.tight_layout()
    plt.show()

# --- Core ODE System for Expander-Repression Model ---
def g_target(E, E_cut, E_end):
    return np.where(E < E_cut, 1.0,
           np.where(E < E_end, (E_end - E) / (E_end - E_cut), 0.0))

def s_C(C, C_swap):
    return 1 / (1 + np.exp((C - C_swap)))

def C_fun(A, E, C0):
    return C0 / np.sinh(A / E)

def ode_system(t, y, params):
    A, g, E = y
    tau_A, tau_g, g_0, tau_E, E_cut, E_end, k, C_swap, C0 = params

    C = C_fun(A, E, C0)
    g_t = g_0*g_target(E, E_cut, E_end)
    s_val = s_C(C, C_swap)

    dA_dt = (A * g) / tau_A
    dg_dt = (g_t - g) / tau_g
    dE_dt = (-k * E + s_val) / tau_E

    return [dA_dt, dg_dt, dE_dt]

# --- Simulate prior predictive ODE trajectories ---
def simulate_prior_predictive_ODE(t_vals, df: pd.DataFrame, n_samples=50):
    np.random.seed(42)

    plt.figure(figsize=(10, 6))

    max_A = -np.inf
    worst_params = None
    for _ in range(n_samples):
        # Sample plausible prior values
        tau_A = np.random.lognormal(mean=0.5, sigma=0.3)
        tau_g = np.random.lognormal(mean=0.5, sigma=0.3)
        tau_E = np.random.lognormal(mean=0.5, sigma=0.3)

        g_max = np.random.normal(0, 0.1)

        E_cut = np.random.lognormal(mean=np.log(0.35), sigma=0.3)
        E_end = np.random.lognormal(mean=np.log(1.5), sigma=0.3)

        k = np.random.lognormal(mean=np.log(0.3), sigma=0.3)
        C_swap = np.random.lognormal(mean=np.log(1.0), sigma=0.3)
        C0 = np.random.lognormal(mean=np.log(3.0), sigma=0.3)

        A0 = np.random.lognormal(mean=np.log(1.0), sigma=0.3) 
        g0 = np.random.normal(0.2, 0.2)
        E0 = np.random.lognormal(mean=np.log(0.2), sigma=0.3) 

        params = [tau_A, tau_g, g_max, tau_E, E_cut, E_end, k, C_swap, C0]

        sol = solve_ivp(
            lambda t, y: ode_system(t, y, params),
            [t_vals[0], t_vals[-1]],
            [A0, g0, E0],
            t_eval=t_vals,
            method="RK45",
            max_step=1.0
        )
        if sol.success:
            current_max_A = np.max(sol.y[0])
            if current_max_A > max_A:
                max_A = current_max_A
                worst_params = params

        if sol.success:
            plt.plot(sol.t, sol.y[0], color="gray", alpha=0.3)
    worst_params_dict = {
        "tau_A": worst_params[0],
        "tau_g": worst_params[1],
        "g_0": worst_params[2],
        "tau_E": worst_params[3],
        "E_cut": worst_params[4],
        "E_end": worst_params[5],
        "k": worst_params[6],
        "C_swap": worst_params[7],
        "C0": worst_params[8],
        "max_A": max_A
    }

    print(worst_params_dict)

    sns.scatterplot(data=df, x="time in hpf", y="Surface Area", hue="condition", palette="tab10")
    plt.title("Prior Predictive Trajectories vs Observed Data")
    plt.xlabel("Time (hpf)")
    plt.ylabel("Surface Area A(t)")
    plt.tight_layout()
    plt.show()

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

def compile_and_fit_stan_model(model_path: str, data_dict: dict):
    model = CmdStanModel(stan_file=model_path)
    fit = model.sample(data=data_dict, chains=4, iter_warmup=3000, iter_sampling=1000, show_progress=True)
    return fit

def posterior_diagnostics(fit, data_dict):
    az_data = az.from_cmdstanpy(posterior=fit, posterior_predictive=["A_Dev_ppc", "A_Reg_ppc"])
    samples = az_data
    f_ppc_dev = samples.posterior_predictive.A_Dev_ppc.stack({"sample": ("chain", "draw")}).transpose("sample", "A_Dev_ppc_dim_0")
    f_ppc_reg = samples.posterior_predictive.A_Reg_ppc.stack({"sample": ("chain", "draw")}).transpose("sample", "A_Reg_ppc_dim_0")


    output_file("corner_Dev.html")
    save(corner(samples, parameters=["A_end", "A_cut", "A_0_Dev", "g_0_Dev", "alpha", "beta_"], xtick_label_orientation=np.pi/4))

    output_file("corner_Reg.html")
    save(corner(samples, parameters=["A_end", "A_cut", "A_0_Reg", "g_0_Reg", "alpha", "beta_"], xtick_label_orientation=np.pi/4))

    output_file("ppc_Dev.html")
    save(predictive_regression(
        f_ppc_dev,
        samples_x=np.array(data_dict["t_ppc"]),
        data=np.dstack((data_dict["t_Dev"], data_dict["A_Dev"])).squeeze(),
        x_axis_label='t',
        y_axis_label='A'))

    output_file("ppc_Reg.html")
    save(predictive_regression(
        f_ppc_reg,
        samples_x=np.array(data_dict["t_ppc"]),
        data=np.dstack((data_dict["t_Reg"], data_dict["A_Reg"])).squeeze(),
        x_axis_label='t',
        y_axis_label='A'))

    # Save posterior samples to CSV
    posterior = samples.posterior
    results = {var_name: posterior[var_name].values.flatten() for var_name in posterior.data_vars}
    results_df = pd.DataFrame(results)
    fit_results_path = os.path.join(scalar_path, "fit_results")
    os.makedirs(fit_results_path, exist_ok=True)
    results_df.to_csv(os.path.join(fit_results_path, "area_sampled_parameter_results_exp_rep.csv"), index=False)

# --- Main workflow ---
def main():
    df = getData()
    df = df[~df['condition'].isin(['4850cut', '7230cut'])]
    df['Surface Area'] = df['Surface Area'] / 10000
    df = df[["time in hpf", "Surface Area", "condition"]].dropna()

    explorative_plotting(df)

    t_range = np.linspace(df["time in hpf"].min(), df["time in hpf"].max(), 100)
    simulate_prior_predictive_ODE(t_range, df)
    return
    data_dict = prepare_stan_data(df)

    model_path = "3_fits/fit_exp_rep.stan"
    fit = compile_and_fit_stan_model(model_path, data_dict)
    posterior_diagnostics(fit,data_dict)


if __name__ == "__main__":
    main()
