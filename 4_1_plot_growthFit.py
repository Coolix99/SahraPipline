import pandas as pd
import os
import numpy as np

from plotHelper import plot_scatter_corner,plot_double_timeseries_II,add_data_to_plot_II,add_fit_to_plot_II
from bokeh.io import show
from bokeh.plotting import figure

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

def A_theor_Dose(t, A_0, g_0,D_0, C0,  C1,  D_decrease,  g_max,  tau_g):
    t48 = t - 48

    def ode_system(t, y):
        A, D, g = y

        dAdt = g * A
        dDdt = C0 + C1 * A
         
        g_theory=0.0
        D_end = 10 + D_decrease
        if (D < 10): 
            g_theory = g_max
        elif (D < D_end):
            g_theory = g_max * (D_end - D) / (D_end - 10)

        dgdt = -(g - g_theory)/tau_g
        return [dAdt,dDdt, dgdt]

    # Initial conditions
    y0 = [A_0,D_0, g_0]

    # Solve the ODE for A and g
    solution = solve_ivp(ode_system, [t48.min(), t48.max()], y0, t_eval=t48, method='RK45')
    
    A_values = solution.y[0]  # A(t)
    D_values = solution.y[1]
    g_values = solution.y[2]  # g(t)
    
    return A_values,D_values, g_values


def getPosterior_setPoint(max_samples=None):
    csv_file_path = os.path.join(Curv_Thick_path, "area_sampled_parameter_results_setPoint.csv")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Limit to max_samples if specified
    if max_samples is not None:
        df = df.head(max_samples)
    return df

def getPosterior_Dose(max_samples=None):
    csv_file_path = os.path.join(Curv_Thick_path, "area_sampled_parameter_results_Dose.csv")

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

def getTrajectories_Dose(df,t0=48,t_end=144):

    # Placeholder for results
    results = {
        'A': [],
        'D': [],
        'A_noisy': [],
        'g': [],
    }

    # Time vector for solving ODE
    t_values = np.linspace(t0, t_end, 100)  # Example time values from 0 to 100

    # Loop over each row in the DataFrame (parameter set)
    for idx, row in df.iterrows():
        # Extract parameters for Development
        A_0, g_0, D_0, C0, C1, D_decrease, g_max , tau_g = row[['A_0', 'g_0','D_0','C0','C1','D_decrease','g_max', 'tau_g']]
        
        # Calculate theoretical and noisy values for Development
        A,D, g = A_theor_Dose(t_values, A_0, g_0,D_0, C0,  C1,  D_decrease,  g_max,  tau_g)
        A_noisy = A + np.random.normal(0, row['sigma'], len(A))  # Add noise

        # Add Development results
        results['A'].append(A)
        results['D'].append(D)
        results['A_noisy'].append(A_noisy)
        results['g'].append(g)


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

def make_FFcut_Parameters_setPoint(df_Posterior_setPoint):
    selected_columns = ['g_0_Reg','A_cut','A_end','alpha','beta_','sigma']
    df_1=df_Posterior_setPoint[selected_columns].rename(columns={'g_0_Reg':'g_0'})
    df_1['A_0'] = np.random.normal(loc=3.5, scale=0.75, size=len(df_1))
    
    df_2=df_1.copy()
    df_2['g_0']=df_2['beta_']*(df_2['A_end']-df_2['A_0'])/df_2['A_end']
    
    return df_1,df_2

def make_4850cut_Parameters_setPoint(df_Posterior_setPoint): 
    selected_columns = ['g_0_Reg','A_cut','A_end','alpha','beta_','sigma','A_0_Dev']
    df_1=df_Posterior_setPoint[selected_columns].rename(columns={'g_0_Reg':'g_0','A_0_Dev':'A_0'})
    df_1['A_0'] = df_1['A_0']*0.4
    
    df_2=df_1.copy()
    df_2['g_0']=df_2['beta_']*(df_2['A_end']-df_2['A_0'])/df_2['A_end']
    
    return df_1,df_2


def make_FFcut_Parameters_Dose(df_Posterior_setPoint):
    selected_columns = ['g_0_Reg','C0','C1','D_decrease','g_max', 'tau_g','sigma','D_0_Reg']
    df_1=df_Posterior_setPoint[selected_columns].rename(columns={'g_0_Reg':'g_0', 'D_0_Reg':'D_0'})
    df_1['A_0'] = np.random.normal(loc=3.5, scale=0.75, size=len(df_1))
    
    df_2=df_1.copy()
    df_2['D_0']=df_2['D_decrease']/3
    
    return df_1,df_2

def make_4850cut_Parameters_Dose(df_Posterior_setPoint):
    selected_columns = ['A_0_Dev','g_0_Reg','C0','C1','D_decrease','g_max', 'tau_g','sigma','D_0_Reg']
    df_1=df_Posterior_setPoint[selected_columns].rename(columns={'A_0_Dev':'A_0','g_0_Reg':'g_0', 'D_0_Reg':'D_0'})
    
    df_1['A_0'] = df_1['A_0']*0.4
    
    df_2=df_1.copy()
    df_2['D_0']=0
    
    return df_1,df_2




def main():
    df=getData()
    p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    p=add_data_to_plot_II(df,p,y_col='Surface Area',category='4850cut',y_scaling=1e-4,color='black',width=width)
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

    df_FFcut_1,df_FFcut_2=make_FFcut_Parameters_setPoint(df_Posterior_setPoint)
    fit_FFcut_1,t_FFcut_1=getTrajectories_setPoint(df_FFcut_1,t0=72)
    fit_FFcut_2,t_FFcut_2=getTrajectories_setPoint(df_FFcut_2,t0=72)
    fit_results_FFcut_1_setPoint = {
        't_values': t_FFcut_1,
        'fits': fit_FFcut_1['A_noisy']
    }
    fit_results_FFcut_2_setPoint = {
        't_values': t_FFcut_2,
        'fits': fit_FFcut_2['A_noisy']
    }

    df_4850cut_1,df_4850cut_2=make_4850cut_Parameters_setPoint(df_Posterior_setPoint)
    fit_4850cut_1,t_4850cut_1=getTrajectories_setPoint(df_4850cut_1,t_end=168)
    fit_4850cut_2,t_4850cut_2=getTrajectories_setPoint(df_4850cut_2,t_end=168)
    fit_results_4850cut_1_setPoint = {
        't_values': t_4850cut_1,
        'fits': fit_4850cut_1['A_noisy']
    }
    fit_results_4850cut_2_setPoint = {
        't_values': t_4850cut_2,
        'fits': fit_4850cut_2['A_noisy']
    }
    
    
    
    ############Dose model fitted with Reg Dev Area##########################################

    df_Posterior_Dose=getPosterior_Dose(max_samples)
    print(df_Posterior_Dose)
    
    selected_columns = ['A_0_Dev', 'g_0_Dev','C0','C1','D_decrease','g_max', 'tau_g','sigma']
    rename_map = {'A_0_Dev': 'A_0', 'g_0_Dev': 'g_0'}
    df_Dev_Dose = df_Posterior_Dose[selected_columns].rename(columns=rename_map)
    df_Dev_Dose['D_0']=0
    fit_Dev,t_Dev=getTrajectories_Dose(df_Dev_Dose,t_end=168)
    # closest_index = np.abs(t_Dev - 72).argmin()
    # for D_arr in fit_Dev['D']:
    #     print(D_arr[closest_index])
    fit_results_dev_Dose = {
        't_values': t_Dev,
        'fits': fit_Dev['A_noisy']
    }

    selected_columns = ['A_0_Reg', 'g_0_Reg','C0','C1','D_decrease','g_max', 'tau_g','sigma','D_0_Reg']
    rename_map = {'A_0_Reg': 'A_0', 'g_0_Reg': 'g_0','D_0_Reg':'D_0'}
    df_Reg_Dose = df_Posterior_Dose[selected_columns].rename(columns=rename_map)
    fit_Reg,t_Reg=getTrajectories_Dose(df_Reg_Dose,t_end=168)
    fit_results_reg_Dose = {
        't_values': t_Reg,
        'fits': fit_Reg['A_noisy']
    }

    df_FFcut_1,df_FFcut_2=make_FFcut_Parameters_Dose(df_Posterior_Dose)
    fit_FFcut_1,t_FFcut_1=getTrajectories_Dose(df_FFcut_1,t0=72)
    fit_FFcut_2,t_FFcut_2=getTrajectories_Dose(df_FFcut_2,t0=72)
    fit_results_FFcut_1_Dose = {
        't_values': t_FFcut_1,
        'fits': fit_FFcut_1['A_noisy']
    }
    fit_results_FFcut_2_Dose = {
        't_values': t_FFcut_2,
        'fits': fit_FFcut_2['A_noisy']
    }
    
    df_4850cut_1,df_4850cut_2=make_4850cut_Parameters_Dose(df_Posterior_Dose)
    fit_4850cut_1,t_4850cut_1=getTrajectories_Dose(df_4850cut_1,t_end=168)
    fit_4850cut_2,t_4850cut_2=getTrajectories_Dose(df_4850cut_2,t_end=168)
    fit_results_4850cut_1_Dose = {
        't_values': t_4850cut_1,
        'fits': fit_4850cut_1['A_noisy']
    }
    fit_results_4850cut_2_Dose = {
        't_values': t_4850cut_2,
        'fits': fit_4850cut_2['A_noisy']
    }
    
    ##########Plot############
    #FFcut
    print(df.columns)
    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    # show(p)
    # p = add_fit_to_plot_II(p, fit_results_dev_setPoint, color='#5056fa',label='Development (SP)')  
    # p = add_fit_to_plot_II(p, fit_results_reg_setPoint, color='#fac150',label='Regeneration (SP)')
    # p = add_fit_to_plot_II(p, fit_results_dev_Dose, color='#02067a',label='Development (D)')  
    # p = add_fit_to_plot_II(p, fit_results_reg_Dose, color='#855901',label='Regeneration (D)')
    # show(p)
    # p = add_fit_to_plot_II(p, fit_results_FFcut_1_setPoint, color='#72fc79',label='FFcut (SP I)')
    # p = add_fit_to_plot_II(p, fit_results_FFcut_2_setPoint, color='#017d07',label='FFcut (SP II)')
    # p = add_fit_to_plot_II(p, fit_results_FFcut_1_Dose, color='#c253f5',label='FFcut (D I)')
    # p = add_fit_to_plot_II(p, fit_results_FFcut_2_Dose, color='#570080',label='FFcut (D II)')
    # show(p)
    # p=add_data_to_plot_II(df,p,y_col='Surface Area',category='72FF_cut',y_scaling=1e-4,color='black',width=width)
    # show(p)

    
    p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    show(p)
    p = add_fit_to_plot_II(p, fit_results_dev_setPoint, color='#5056fa',label='Development (SP)')  
    p = add_fit_to_plot_II(p, fit_results_reg_setPoint, color='#fac150',label='Regeneration (SP)')
    p = add_fit_to_plot_II(p, fit_results_dev_Dose, color='#02067a',label='Development (D)')  
    p = add_fit_to_plot_II(p, fit_results_reg_Dose, color='#855901',label='Regeneration (D)')
    show(p)
    p = add_fit_to_plot_II(p, fit_results_4850cut_1_setPoint, color='#72fc79',label='4850cut (SP I)')
    p = add_fit_to_plot_II(p, fit_results_4850cut_2_setPoint, color='#017d07',label='4850cut (SP II)')
    p = add_fit_to_plot_II(p, fit_results_4850cut_1_Dose, color='#c253f5',label='4850cut (D I)')
    p = add_fit_to_plot_II(p, fit_results_4850cut_2_Dose, color='#570080',label='4850cut (D II)')
    show(p)
    p=add_data_to_plot_II(df,p,y_col='Surface Area',category='4850cut',y_scaling=1e-4,color='black',width=width)
    show(p)
    return

def plot_fit_and_derivative():
    df=getData()
    p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
   
    show(p)
    
    max_samples=None
    ############Set Point model fitted with Reg Dev Area##########################################
    df_Posterior_setPoint=getPosterior_setPoint(max_samples)

    selected_columns = ['A_0_Dev', 'g_0_Dev','A_cut','A_end','alpha','beta_','sigma']
    rename_map = {'A_0_Dev': 'A_0', 'g_0_Dev': 'g_0'}
    df_Dev_setPoint = df_Posterior_setPoint[selected_columns].rename(columns=rename_map)
    fit_Dev,t_Dev=getTrajectories_setPoint(df_Dev_setPoint,t_end=144)
    fit_results_dev_setPoint = {
        't_values': t_Dev,
        'fits': fit_Dev['A_noisy']
    }
    fit_results_dev_setPoint_g = {
        't_values': t_Dev,
        'fits': fit_Dev['g']
    }

    selected_columns = ['A_0_Reg', 'g_0_Reg','A_cut','A_end','alpha','beta_','sigma']
    rename_map = {'A_0_Reg': 'A_0', 'g_0_Reg': 'g_0'}
    df_Reg_setPoint = df_Posterior_setPoint[selected_columns].rename(columns=rename_map)
    fit_Reg,t_Reg=getTrajectories_setPoint(df_Reg_setPoint,t_end=144)
    fit_results_reg_setPoint = {
        't_values': t_Reg,
        'fits': fit_Reg['A_noisy']
    }
    fit_results_reg_setPoint_g = {
        't_values': t_Reg,
        'fits': fit_Reg['g']
    }

    ############Dose model fitted with Reg Dev Area##########################################

    df_Posterior_Dose=getPosterior_Dose(max_samples)
    print(df_Posterior_Dose)
    
    selected_columns = ['A_0_Dev', 'g_0_Dev','C0','C1','D_decrease','g_max', 'tau_g','sigma']
    rename_map = {'A_0_Dev': 'A_0', 'g_0_Dev': 'g_0'}
    df_Dev_Dose = df_Posterior_Dose[selected_columns].rename(columns=rename_map)
    df_Dev_Dose['D_0']=0
    fit_Dev,t_Dev=getTrajectories_Dose(df_Dev_Dose,t_end=144)
   
    fit_results_dev_Dose = {
        't_values': t_Dev,
        'fits': fit_Dev['A_noisy']
    }
    fit_results_dev_Dose_g = {
        't_values': t_Dev,
        'fits': fit_Dev['g']
    }


    selected_columns = ['A_0_Reg', 'g_0_Reg','C0','C1','D_decrease','g_max', 'tau_g','sigma','D_0_Reg']
    rename_map = {'A_0_Reg': 'A_0', 'g_0_Reg': 'g_0','D_0_Reg':'D_0'}
    df_Reg_Dose = df_Posterior_Dose[selected_columns].rename(columns=rename_map)
    fit_Reg,t_Reg=getTrajectories_Dose(df_Reg_Dose,t_end=144)
    fit_results_reg_Dose = {
        't_values': t_Reg,
        'fits': fit_Reg['A_noisy']
    }
    fit_results_reg_Dose_g = {
        't_values': t_Reg,
        'fits': fit_Reg['g']
    }


   
    
    ##########Plot############
    print(df.columns)
    p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    p = add_fit_to_plot_II(p, fit_results_dev_setPoint, color='#5056fa',label='Development (SP)')  
    p = add_fit_to_plot_II(p, fit_results_reg_setPoint, color='#fac150',label='Regeneration (SP)')
    p = add_fit_to_plot_II(p, fit_results_dev_Dose, color='#02067a',label='Development (D)')  
    p = add_fit_to_plot_II(p, fit_results_reg_Dose, color='#855901',label='Regeneration (D)')
    show(p)
   
    p = figure(title=f"relative Surface Area growth rate",
               x_axis_label=r'time in hpf',
               y_axis_label=r'dArea/dt/Area $$1/h$$',
               width=700, height=400,
               tools="pan,wheel_zoom,box_zoom,reset,save")
    p = add_fit_to_plot_II(p, fit_results_dev_setPoint_g, color='#5056fa',label='Development (SP)')  
    p = add_fit_to_plot_II(p, fit_results_reg_setPoint_g, color='#fac150',label='Regeneration (SP)')  
    p = add_fit_to_plot_II(p, fit_results_dev_Dose_g, color='#02067a',label='Development (D)')  
    p = add_fit_to_plot_II(p, fit_results_reg_Dose_g, color='#855901',label='Regeneration (D)')
    show(p)

    return


if __name__ == "__main__":
    plot_fit_and_derivative()
    #main()