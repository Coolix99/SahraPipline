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
    sns.scatterplot(data=df, x="time in hpf", y="Surface Area", hue="condition")
    plt.title("Surface Area over Time by Condition")
    plt.xlabel("Time (hpf)")
    plt.ylabel("Surface Area")
    plt.tight_layout()
    plt.show()

def ode_system(t, y, alpha, beta_, A_end, A_cut):
    A, g = y
    dA_dt = g * A
    if A < A_cut:
        dg_dt = -alpha * (g - beta_ * (A_end - A_cut) / A_end)
    else:
        dg_dt = -alpha * (g - beta_ * (A_end - A) / A_end)
    return [dA_dt, dg_dt]

def g_A_piecewise(A, beta_, A_end, A_cut):
    if A < A_cut:
        return beta_ * (A_end - A_cut) / A_end
    else:
        return beta_ * (A_end - A) / A_end

def ode_simplified(t, y, beta_, A_end, A_cut):
    A = y[0]
    g = g_A_piecewise(A, beta_, A_end, A_cut)
    return [A * g]

def simulate_prior_predictive_ODE(t_vals, df: pd.DataFrame, n_samples=50):
    np.random.seed(42)
    plt.figure(figsize=(10, 6))

    for _ in range(n_samples):
        beta_tilde = np.random.normal(-1.0, 0.5)
        A_end_tilde = np.random.normal(1.0, 0.1)
        A_cut_tilde = np.random.normal(0, 1)
        A_0_tilde = np.random.normal(0.3, 0.15)

        beta_ = 10 ** beta_tilde
        A_end = 10 ** A_end_tilde
        A_cut = 2 + 4 * (1 / (1 + np.exp(-A_cut_tilde)))
        A_0 = 10 ** A_0_tilde

        sol = solve_ivp(
            lambda t, y: ode_simplified(t, y, beta_, A_end, A_cut),
            [t_vals[0], t_vals[-1]],
            [A_0],
            t_eval=t_vals,
            method="RK45"
        )

        plt.plot(sol.t, sol.y[0], color="gray", alpha=0.3)

    sns.scatterplot(data=df, x="time in hpf", y="Surface Area", hue="condition", palette="tab10")
    plt.title("Simplified Prior Predictive Trajectories vs Observed Data")
    plt.xlabel("Time (hpf)")
    plt.ylabel("Surface Area A(t)")
    plt.tight_layout()
    plt.show()

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
    save(corner(samples, parameters=["A_end", "A_cut", "A_0_Dev", "beta_"], xtick_label_orientation=np.pi/4))

    output_file("corner_Reg.html")
    save(corner(samples, parameters=["A_end", "A_cut", "A_0_Reg", "beta_"], xtick_label_orientation=np.pi/4))


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
    results_df.to_csv(os.path.join(fit_results_path, "area_sampled_parameter_results_delayless.csv"), index=False)

    sigma_samples = samples.posterior["sigma"].values.flatten()
    sigma_mean = np.mean(sigma_samples)
    sigma_std = np.std(sigma_samples)
    print(f"Posterior sigma: mean = {sigma_mean:.4f}, std = {sigma_std:.4f}")

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

def main():
    df = getData()
    df['Surface Area'] = df['Surface Area'] / 10000
    df = df[~df['condition'].isin(['4850cut', '7230cut'])]

    df = df[["time in hpf", "Surface Area", "condition"]].dropna()

    explorative_plotting(df)
    t_range = np.linspace(df["time in hpf"].min(), df["time in hpf"].max(), 100)
    simulate_prior_predictive_ODE(t_range,df)
    
    data_dict = prepare_stan_data(df)

    model_path = "3_fits/fit_delayless.stan"
    fit = compile_and_fit_stan_model(model_path, data_dict)
    posterior_diagnostics(fit,data_dict)


if __name__ == "__main__":
    main()
