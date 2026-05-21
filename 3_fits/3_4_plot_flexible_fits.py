import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from bokeh.io import show
from plotHelper.bokeh_timeseries_plot import plot_double_timeseries
from plotHelper.plotBokehHelper_old import add_fit_to_plot_II
from utilsScalar import getData, scalar_path

# --- Model configurations ---
model_config = {
    "setPoint": {
        "filename":"area_sampled_parameter_results_setPoint.csv",
        "param_map_dev": {'A_0_Dev': 'A_0', 'g_0_Dev': 'g_0'},
        "param_map_reg": {'A_0_Reg': 'A_0', 'g_0_Reg': 'g_0'},
        "required_columns": ['A_0', 'g_0', 'A_cut', 'A_end', 'alpha', 'beta_', 'sigma'],
        "ode_solver": lambda t, p: solve_ivp(
            lambda t, y: [y[1]*y[0],
                          -p['alpha']*(y[1] - p['beta_']*(p['A_end'] - p['A_cut'])/p['A_end'])
                          if y[0] < p['A_cut'] else
                          -p['alpha']*(y[1] - p['beta_']*(p['A_end'] - y[0])/p['A_end'])],
            [t[0], t[-1]], [p['A_0'], p['g_0']], t_eval=t
        )
    },
    "linear": {
        "filename":"area_sampled_parameter_results_delayless_lin.csv",
        "param_map_dev": {'A_0_Dev': 'A_0'},
        "param_map_reg": {'A_0_Reg': 'A_0'},
        "required_columns": ['A_0', 'A_end', 'beta_', 'sigma'],
        "ode_solver": lambda t, p: solve_ivp(
            lambda t, y: [y[0] * (p['beta_']*(p['A_end'] - y[0])/p['A_end'])],
            [t[0], t[-1]], [p['A_0']], t_eval=t
        )
    },
    "delayless": {
        "filename":"area_sampled_parameter_results_delayless_full.csv",
        "param_map_dev": {'A_0_Dev': 'A_0'},
        "param_map_reg": {'A_0_Reg': 'A_0'},
        "required_columns": ['A_0', 'A_cut', 'A_end', 'beta_', 'sigma'],
        "ode_solver": lambda t, p: solve_ivp(
            lambda t, y: [y[0] * (p['beta_']*(p['A_end'] - p['A_cut'])/p['A_end']
                                  if y[0] < p['A_cut'] else
                                  p['beta_']*(p['A_end'] - y[0])/p['A_end'])],
            [t[0], t[-1]], [p['A_0']], t_eval=t
        )
    },
"setPoint_steadyg": {

    "filename":
        "area_sampled_parameter_results_setPoint_linear_all.csv",

    "conditions": {

        "Development": {
            "param_map": {
                "A_0_Dev": "A_0",
                "sigma_Dev": "sigma",
            },
            "steady_state_init": True,
            "color": "#5056fa",
        },

        "Regeneration": {
            "param_map": {
                "A_0_Reg": "A_0",
                "sigma_Reg": "sigma",
            },
            "steady_state_init": False,
            "color": "#fac150",
        },

        "4850cut": {
            "param_map": {
                "A_0_4850cut": "A_0",
                "sigma_4850cut": "sigma",
            },
            "steady_state_init": False,
            "color": "#5b3a2a",
        },

        "7230cut": {
            "param_map": {
                "A_0_7230cut": "A_0",
                "sigma_7230cut": "sigma",
            },
            "steady_state_init": False,
            "color": "#d8b07a",
            "t0": 72,
        },

        "Smoc12_Dev": {
            "param_map": {
                "A_0_smoc_dev": "A_0",
                "sigma_smoc_dev": "sigma",
                "A_end_smocdev": "A_end",
            },
            "steady_state_init": True,
            "color": "#7b5ea7",
        },

        "smoc_reg": {
            "param_map": {
                "A_0_smoc_reg": "A_0",
                "sigma_smoc_reg": "sigma",
            },
            "steady_state_init": False,
            "color": "#df3066",
        },
    },

    "required_columns": [
        "A_0",
        "A_end",
        "alpha",
        "beta_",
        "sigma",
    ],

    "ode_solver": lambda t, p: solve_ivp(
        lambda t, y: [
            y[1] * y[0],
            -p["alpha"] * (
                y[1]
                - p["beta_"] * (p["A_end"] - y[0]) / p["A_end"]
            )
        ],

        [t[0], t[-1]],

        [
            p["A_0"],

            p["beta_"] * (
                p["A_end"] - p["A_0"]
            ) / p["A_end"]

            if p["steady_state_init"]

            else 0.0
        ],

        t_eval=t
    )
},
}

def getPosterior(filename: str, max_samples=None):
    df = pd.read_csv(os.path.join(scalar_path, 'fit_results', filename))
    return df.head(max_samples) if max_samples else df

def getTrajectories(df, model_name, condition, t0=48, t_end=144):
    config = model_config[model_name]
    results = {'A': [], 'A_noisy': []}
    t = np.linspace(t0, t_end, 100)
    for _, row in df.iterrows():
        params = {k: row[k] for k in config['required_columns']}

        params["steady_state_init"] = (
            config["conditions"][condition]["steady_state_init"]
        )

        sol = config['ode_solver'](t, params)
        A = sol.y[0] if sol.y.ndim > 1 else sol.y
        results['A'].append(A)
        results['A_noisy'].append(A + np.random.normal(0, params['sigma'], len(A)))
    return results, t

def prepare_df(df_posterior, model_name, param_map):
    raw_cols = list(param_map.keys())
    extra_cols = [c for c in model_config[model_name]['required_columns'] if c not in param_map.values()]
    return df_posterior[raw_cols + extra_cols].rename(columns=param_map)


def plot_model(model_name="setPoint", max_samples=None):
    df = getData()
    df = df[~df['condition'].isin(['4850cut', '7230cut'])]
    df = df[['time in hpf', 'Surface Area', 'condition']].dropna()
    
    categories = {'Development': 'blue', 'Regeneration': 'orange'}
    p, _ = plot_double_timeseries(
        df, categories=categories, y_col='Surface Area',
        style='box', y_scaling=1e-4, y_name=r'Area $$(100 \mu m)^2$$',
        test_significance=True, y0=0
    )

    df_post = getPosterior(model_config[model_name]['filename'], max_samples)

    config = model_config[model_name]

    for condition, cond_cfg in config["conditions"].items():

        df_cond = prepare_df(
            df_post,
            model_name,
            cond_cfg["param_map"]
        )

        fit, t_fit = getTrajectories(
            df_cond,
            model_name,
            condition=condition,
            t0=cond_cfg.get("t0", 48),
            t_end=168,
        )

        result = {
            "t_values": t_fit,
            "fits": fit["A_noisy"]
        }

        p = add_fit_to_plot_II(
            p,
            result,
            color=cond_cfg["color"],
            label=f"{condition} ({model_name})"
        )
    show(p)

if __name__ == "__main__":
    #plot_model(model_name="delayless")  
    #plot_model(model_name="setPoint")  
    # plot_model(model_name="linear")  
    plot_model(model_name="setPoint_steadyg")