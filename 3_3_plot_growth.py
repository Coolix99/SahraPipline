import pandas as pd
import os
import numpy as np

from config import *

#from plotMatPlotHelper import 
from plotBokehHelper import plot_single_timeseries,plot_double_timeseries,plot_scatter,plot_explanation

def getData():
    hdf5_file_path = os.path.join(Curv_Thick_path, 'scalarGrowthData.h5')

    # Load the DataFrame from the HDF5 file
    df = pd.read_hdf(hdf5_file_path, key='data')

    # Calculate new columns:
    # L AP * L PD
    df['L AP * L PD'] = df['L AP'] * df['L PD']

    # V / A (Volume / Surface Area)
    df['V / A'] = df['Volume'] / df['Surface Area']

    # Int_dA_d / A (Integrated Thickness / Surface Area)
    df['Int_dA_d / A'] = df['Integrated Thickness'] / df['Surface Area']

    df['log L AP'] = np.log(df['L AP'])
    df['log L PD'] = np.log(df['L PD'])


    return df

from scipy.integrate import solve_ivp
rng = np.random.default_rng()
def A_theor(t, A_0, g_0, alpha, beta, A_end, A_cut):
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

def getFit(max_samples=None):
    csv_file_path = os.path.join(Curv_Thick_path, "area_sampled_parameter_results.csv")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Limit to max_samples if specified
    if max_samples is not None:
        df = df.head(max_samples)

    # Placeholder for results
    results = {
        'A_Development': [],
        'A_Development_noisy': [],
        'g_Development': [],
        'A_Regeneration': [],
        'A_Regeneration_noisy': [],
        'g_Regeneration': []
    }

    # Time vector for solving ODE
    t_values = np.linspace(48, 144, 100)  # Example time values from 0 to 100

    # Loop over each row in the DataFrame (parameter set)
    for idx, row in df.iterrows():
        # Extract parameters for Development
        A_0_Dev, g_0_Dev, alpha_Dev, beta_Dev, A_end_Dev, A_cut_Dev = row[['A_0_Dev', 'g_0_Dev', 'alpha', 'beta_', 'A_end_Dev', 'A_cut_Dev']]
        
        # Calculate theoretical and noisy values for Development
        A_Dev, g_Dev = A_theor(t_values, A_0_Dev, g_0_Dev, alpha_Dev, beta_Dev, A_end_Dev, A_cut_Dev)
        A_Dev_noisy = A_Dev + np.random.normal(0, row['sigma'], len(A_Dev))  # Add noise

        # Add Development results
        results['A_Development'].append(A_Dev)
        results['A_Development_noisy'].append(A_Dev_noisy)
        results['g_Development'].append(g_Dev)

        # Extract parameters for Regeneration
        A_0_Reg, g_0_Reg, alpha_Reg, beta_Reg, A_end_Reg, A_cut_Reg = row[['A_0_Reg', 'g_0_Reg', 'alpha', 'beta_', 'A_end_Reg', 'A_cut_Reg']]
        
        # Calculate theoretical and noisy values for Regeneration
        A_Reg, g_Reg = A_theor(t_values, A_0_Reg, g_0_Reg, alpha_Reg, beta_Reg, A_end_Reg, A_cut_Reg)
        A_Reg_noisy = A_Reg + np.random.normal(0, row['sigma'], len(A_Reg))  # Add noise

        # Add Regeneration results
        results['A_Regeneration'].append(A_Reg)
        results['A_Regeneration_noisy'].append(A_Reg_noisy)
        results['g_Regeneration'].append(g_Reg)

    return results,t_values

def plot_results(results, t_values):
    import matplotlib.pyplot as plt

    # Plotting Development and Regeneration results

    plt.figure(figsize=(10, 6))



    # Plot Development

    for idx, A_Dev in enumerate(results['A_Development']):

        plt.plot(t_values, A_Dev, label=f'Development Set {idx+1}', linestyle='--', color='blue')



    # Plot Regeneration

    for idx, A_Reg in enumerate(results['A_Regeneration']):

        plt.plot(t_values, A_Reg, label=f'Regeneration Set {idx+1}', linestyle='-', color='orange')



    # Add labels and legend

    plt.xlabel('Time (hpf)')

    plt.ylabel('A values')

    plt.title('Theoretical A values for Development and Regeneration')

    plt.legend(loc='best')

    plt.grid(True)

    plt.show()

def main():
    # plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='whiskers', title="Whiskers Explanation")
    # plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='box', title="Box Plot Explanation")
    # plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='violin', title="Violin Plot Explanation")
    results,t_values=getFit()

    df=getData()
    print(df)

    

    #Checks
    # plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='category',show_fit=True,show_div='Residual')
    # plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='time',show_fit=True,show_div='Residual')

    # plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='category',show_fit=True,show_div='Residual')
    # plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='time',show_fit=True,show_div='Residual')

    # plot_scatter(df, x_col='L AP', y_col='L PD', mode='category',show_fit=True,show_div='Residual')
    # plot_scatter(df, x_col='L AP', y_col='L PD', mode='time',show_fit=True,show_div='Residual')

    #plot_scatter(df, x_col='log L PD', y_col='log L AP',x_name=r'$$log(L_{PD})$$',y_name=r'$$log(L_{AP})$$', mode='category',show_fit=True,show_div='Residual')
    #plot_scatter(df, x_col='log L PD', y_col=r'$$log L AP$$', mode='time',show_fit=True,show_div='Residual')

    # #simple_plot(df, filter_col='condition', filter_value='Regeneration', y_col='Volume') #just debuggin

    # plot_single_timeseries(df, filter_col='condition', filter_value='Regeneration', y_col='Volume', style='violin', color='orange',width=None)
    # plot_double_timeseries(df, y_col='Volume', style='violin')
    # plot_double_timeseries(df, y_col='Surface Area', style='box')
    fit={
        't_values':t_values,
        'Development': results['A_Development_noisy'],
        'Regeneration': results['A_Regeneration_noisy']
    }
    plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=False,y0=0,fit_results=fit,show_n=False)
    # plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    #plot_double_timeseries(df, y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$',test_significance=True,y0=0,show_n=False)
    #plot_double_timeseries(df, y_col='V / A', style='violin',y_scaling=1.0,y_name=r'Mean thickness $$(\mu m)$$',test_significance=True,y0=0,show_n=False)



if __name__ == "__main__":
    main()