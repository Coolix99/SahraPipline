import pandas as pd
import os
import numpy as np

from plotHelper.plotBokehHelper_old import plot_scatter_corner,plot_double_timeseries_II,add_data_to_plot_II,add_fit_to_plot_II
from plotHelper.bokeh_timeseries_plot import plot_double_timeseries,add_data_to_plot
from bokeh.io import show
from bokeh.plotting import figure

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

from utilsScalar import getData, scalar_path


from scipy.integrate import solve_ivp
rng = np.random.default_rng()
def A_theor_setPoint(t, A_0, g_0, alpha, beta, A_end, A_cut):
    t48 = t - 48
    def ode_system(t, y):
        A, g = y
        dAdt = g * A
        if A < A_cut:
            dgdt = -alpha * (g - beta * (A_end - A_cut) / A_end)
        else:
            dgdt = -alpha * (g - beta * (A_end - A) / A_end)
        return [dAdt, dgdt]

    # Initial conditions
    y0 = [A_0, g_0]
    # Solve the ODE for A and g
    solution = solve_ivp(ode_system, [t48.min(), t48.max()], y0, t_eval=t48, method='RK45')
    A_values = solution.y[0]  # A(t)
    g_values = solution.y[1]  # g(t)
    
    return A_values, g_values


def getPosterior_setPoint(max_samples=None):
    csv_file_path = os.path.join(scalar_path,'fit_results', "area_sampled_parameter_results_setPoint.csv")
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    # Limit to max_samples if specified
    if max_samples is not None:
        df = df.head(max_samples)
    return df


def getTrajectories_setPoint(df,t0=48,t_end=144):
    # Placeholder for results
    results = {
        'A': [],
        'A_noisy': [],
        'g': [],
    }

    # Time vector for solving ODE
    t_values = np.linspace(t0, t_end, 100)  # Example time values from 0 to 100

    # Loop over each row in the DataFrame (parameter set)
    for idx, row in df.iterrows():
        # Extract parameters for Development
        A_0, g_0, alpha, beta, A_end, A_cut = row[['A_0', 'g_0', 'alpha', 'beta_', 'A_end', 'A_cut']]
        
        # Calculate theoretical and noisy values for Development
        A, g = A_theor_setPoint(t_values, A_0, g_0, alpha, beta, A_end, A_cut)
        A_noisy = A + np.random.normal(0, row['sigma'], len(A))  # Add noise

        # Add Development results
        results['A'].append(A)
        results['A_noisy'].append(A_noisy)
        results['g'].append(g)


    return results,t_values


def main():
    df=getData()

    categories = {'Development': 'blue', 'Regeneration': 'orange'}

    p, width = plot_double_timeseries(
        df,
        categories=categories,
        y_col='Surface Area',
        style='box',
        y_scaling=1e-4,
        y_name=r'Area $$(100 \mu m)^2$$',
        test_significance=True,
        y0=0
    )
    show(p)
    
    
    max_samples=None
    ############Set Point model fitted with Reg Dev Area##########################################
    df_Posterior_setPoint=getPosterior_setPoint(max_samples)

    selected_columns = ['A_0_Dev', 'g_0_Dev','A_cut','A_end','alpha','beta_','sigma']
    rename_map = {'A_0_Dev': 'A_0', 'g_0_Dev': 'g_0'}
    df_Dev_setPoint = df_Posterior_setPoint[selected_columns].rename(columns=rename_map)
    fit_Dev,t_Dev=getTrajectories_setPoint(df_Dev_setPoint,t_end=168)
    fit_results_dev_setPoint = {
        't_values': t_Dev,
        'fits': fit_Dev['A_noisy']
    }

    selected_columns = ['A_0_Reg', 'g_0_Reg','A_cut','A_end','alpha','beta_','sigma']
    rename_map = {'A_0_Reg': 'A_0', 'g_0_Reg': 'g_0'}
    df_Reg_setPoint = df_Posterior_setPoint[selected_columns].rename(columns=rename_map)
    fit_Reg,t_Reg=getTrajectories_setPoint(df_Reg_setPoint,t_end=168)
    fit_results_reg_setPoint = {
        't_values': t_Reg,
        'fits': fit_Reg['A_noisy']
    }

    
    ##########Plot############
    p, width = plot_double_timeseries(
        df,
        categories=categories,
        y_col='Surface Area',
        style='box',
        y_scaling=1e-4,
        y_name=r'Area $$(100 \mu m)^2$$',
        test_significance=True,
        y0=0
    )

    p = add_fit_to_plot_II(p, fit_results_dev_setPoint, color='#5056fa', label='Development (SP)')
    p = add_fit_to_plot_II(p, fit_results_reg_setPoint, color='#fac150', label='Regeneration (SP)')

    show(p)
   
    return


if __name__ == "__main__":
    main()