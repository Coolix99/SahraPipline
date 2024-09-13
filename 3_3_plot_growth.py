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


    return df

from scipy.integrate import solve_ivp
rng = np.random.default_rng()
def A_theor(t, A_0, g_0, alpha, beta, A_end,A_cut):
    t48 = t - 48

    def ode_system(t, y):
        A, g = y
        dAdt = g*A
        if A < A_cut:
            dgdt = -alpha * (g - beta * (A_end - A_cut) / A_end)
        else:
            dgdt = -alpha * (g - beta * (A_end - A) / A_end)
        return [dAdt, dgdt]

    # Initial conditions
    y0 = [A_0, g_0]

    # Solve the ODE
    solution = solve_ivp(ode_system, [t48.min(), t48.max()], y0, t_eval=t48, method='RK45')
    #print("Solver message:", solution.message)
    return solution.y[0]

def linCrit_ppc(x):
    sigma = 0*np.abs(rng.normal(0, 2.0))
    A_0 = 10**rng.normal(0.3, 0.15)
    g_0 = np.abs(rng.normal(0, 0.1))
    alpha = 10**rng.normal(-0.25, 0.25)
    beta = (alpha/4)*10**rng.normal(0, 0.1)
    A_end = 10**rng.normal(1.0, 0.1)
    A_cut = 2 + 4*rng.beta(5.0,5.0)
    #print(A_0, g_0,alpha, beta, A_end)
    return (rng.normal(A_theor(x, A_0, g_0,alpha, beta, A_end,A_cut), sigma)/A_end,
            {"A_0": A_0, "g_0": g_0, "alpha": alpha, "beta": beta, "A_end": A_end, "A_cut":A_cut})

def main():
    # plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='whiskers', title="Whiskers Explanation")
    # plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='box', title="Box Plot Explanation")
    # plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='violin', title="Violin Plot Explanation")
    
    df=getData()
    print(df)

    #Checks
    #plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='time',show_fit=True,show_div=True)
    # plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='time')

    # plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='category')
    # plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='time')


    # #simple_plot(df, filter_col='condition', filter_value='Regeneration', y_col='Volume') #just debuggin

    # plot_single_timeseries(df, filter_col='condition', filter_value='Regeneration', y_col='Volume', style='violin', color='orange',width=None)
    # #plot_double_timeseries(df, y_col='Volume', style='violin')
    # #plot_double_timeseries(df, y_col='Surface Area', style='box')
    plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    plot_double_timeseries(df, y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$',y0=0)
    plot_double_timeseries(df, y_col='V / A', style='violin',y_scaling=1.0,y_name=r'Mean thickness $$(\mu m)$$',y0=0)



if __name__ == "__main__":
    main()