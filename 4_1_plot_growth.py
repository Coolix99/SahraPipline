import pandas as pd
import os
import numpy as np

from plotHelper.plotBokehHelper_old import plot_scatter_corner,plot_double_timeseries_II,add_data_to_plot_II,plot_scatter_II,add_data_scatter_II,add_lines_to_time_series
from bokeh.io import show

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

def getDataED():
    hdf5_file_path = os.path.join(Curv_Thick_path, 'scalarGrowthDataED.h5')

    # Load the DataFrame from the HDF5 file
    df = pd.read_hdf(hdf5_file_path, key='data')

    print(df)


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

def add_ED_Data(df):
    ED_df=getDataED()
    print(ED_df)
    ED_df.rename(columns={'data_name': 'Mask Folder'}, inplace=True)
    merged_df = pd.merge(df, ED_df, on=['Mask Folder', 'condition', 'time in hpf', 'experimentalist', 'genotype'], how='inner')
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].reset_index(drop=True)
    return merged_df

def main():
    # plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='whiskers', title="Whiskers Explanation")
    # plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='box', title="Box Plot Explanation")
    # plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='violin', title="Violin Plot Explanation")
    #results,t_values=getFit()

    df=getData()
    
    
    print(df.columns)
    # filtered_df = df[df['time in hpf'] == 120]

    # # Group by 'condition' and calculate mean and std for specified columns
    # columns_of_interest = ['Volume', 'Surface Area', 'L PD', 'L AP', 'L AP * L PD', 'V / A']
    # grouped_stats = filtered_df.groupby('condition')[columns_of_interest].agg(['mean', 'std'])

    # # Print results
    # print(grouped_stats)
    # raise
    # df.to_csv('output.csv', index=False)  
    # return


    #Checks
    # plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='category',show_fit=True,show_div='Residual')
    # plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='time',show_fit=True,show_div='Residual')

    # plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='category',show_fit=True,show_div='Residual')
    # plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='time',show_fit=True,show_div='Residual')

    # plot_scatter(df, x_col='L AP', y_col='L PD', mode='category',show_fit=True,show_div='Residual')
    # plot_scatter(df, x_col='L AP', y_col='L PD', mode='time',show_fit=True,show_div='Residual')

    #plot_scatter(df, x_col='log L PD', y_col='log L AP',x_name=r'$$log(L_{PD})$$',y_name=r'$$log(L_{AP})$$', mode='category',show_fit=True,show_div='Residual',show_slope_1=True)
    #plot_scatter(df, x_col='log L PD', y_col=r'$$log L AP$$', mode='time',show_fit=True,show_div='Residual')

    # #simple_plot(df, filter_col='condition', filter_value='Regeneration', y_col='Volume') #just debuggin

    # plot_single_timeseries(df, filter_col='condition', filter_value='Regeneration', y_col='Volume', style='violin', color='orange',width=None)
    plot_double_timeseries(df, y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Tissue Volume $$10^6 \mu m^3$$',test_significance=True,y0=0,show_n=False)
    
    plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    plot_double_timeseries(df, y_col='L PD', style='violin',y_scaling=1,y_name=r'$$L_{DP} in \mu m$$',test_significance=True,y0=0)
    plot_double_timeseries(df, y_col='L AP', style='violin',y_scaling=1,y_name=r'$$L_{AP} in \mu m$$',test_significance=True,y0=0)
    plot_double_timeseries(df, y_col='L DV', style='violin',y_scaling=1,y_name=r'$$L_{DV} in \mu m$$',test_significance=True,y0=0)
    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    # p=add_data_to_plot_II(df,p,y_col='Surface Area',category='72FF_cut',y_scaling=1e-4,color='black',width=width)
    # show(p)
    
    df=add_ED_Data(df)
    plot_double_timeseries(df, y_col='Volume ED', style='violin',y_scaling=1e-5,y_name=r'Endosceletal disc Volume $$10^5 \mu m^3$$',test_significance=True,y0=0)
    plot_double_timeseries(df, y_col='N_objects', style='violin',y_name=r'Endosceletal disc Cell number',test_significance=True,y0=0)
    
    # fit={
    #     't_values':t_values,
    #     'Development': results['A_Development_noisy'],
    #     'Regeneration': results['A_Regeneration_noisy']
    # }
    # plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=False,y0=0,fit_results=fit,show_n=False)
    # plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    #plot_double_timeseries(df, y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$',test_significance=True,y0=0,show_n=False)
    #plot_double_timeseries(df, y_col='V / A', style='violin',y_scaling=1.0,y_name=r'Mean thickness $$(\mu m)$$',test_significance=True,y0=0,show_n=False)

    color_dict = {'Regeneration': 'orange',
                 'Development': 'blue', 
                 }
    marker_dict = {'Development': 'circle', 'Regeneration': 'triangle', 'Smoc12': 'square'}
    corner_plot = plot_scatter_corner(df=df, parameters=['Volume','Surface Area','V / A', 'Volume ED','N_objects'], color_col='condition',color_dict=color_dict,marker_col='condition',marker_dict=marker_dict)
    show(corner_plot)
    corner_plot = plot_scatter_corner(df=df, parameters=['Volume','Surface Area','V / A', 'Volume ED','N_objects'], color_col='time in hpf',color_dict=color_dict,marker_col='condition',marker_dict=marker_dict)
    show(corner_plot)

def extract_date_identifier(mask_folder):
    """Extracts date and identifier from the 'Mask Folder' string."""
    parts = mask_folder.split("_")
    date = parts[0]  # First part is always the date
    identifier = parts[-1]  # Last part is always the identifier
    return date, identifier

def group_time_series(df):
    """Groups the Mask Folder entries into time series based on date and identifier."""
    # Extract date and identifier
    df[["date", "identifier"]] = df["Mask Folder"].apply(lambda x: pd.Series(extract_date_identifier(x)))
    
    # Convert date to datetime for proper sorting
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    # Sort by identifier, then by date, then by time in hpf
    df_sorted = df.sort_values(by=["identifier", "date", "time in hpf"])

    time_series = []
    current_series = []

    # Track previous row for comparison
    prev_identifier = None
    prev_time = -float('inf')
    prev_date = None

    for _, row in df_sorted.iterrows():
        # If identifier changes or time goes backward, start a new series
        if row["identifier"] != prev_identifier or row["date"] < prev_date or row["time in hpf"] < prev_time:
            if current_series:
                time_series.append(current_series)
            current_series = []

        # Append current row to series
        current_series.append(row["Mask Folder"])

        # Update tracking variables
        prev_identifier = row["identifier"]
        prev_time = row["time in hpf"]
        prev_date = row["date"]

    # Append the last series
    if current_series:
        time_series.append(current_series)

    return time_series

def new_plots():
    df=getData()
    
    
    print(df.columns)
    print(df.head())
    filtered_df = df[df["condition"] == "4850cut"]
    time_series_list_4850 = group_time_series(filtered_df)

    filtered_df = df[df["condition"] == "7230cut"]
    time_series_list_7230 = group_time_series(filtered_df)
    

    # Output result
    for i, series in enumerate(time_series_list_4850):
        print(f"Time Series {i+1}: {series}")

    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$',test_significance=False,y0=0,show_n=False,base_line_width=2.0,show_quantilles=False)
    # show(p)

    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Surface Area', style='box',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',
    #                                   test_significance=False,y0=0,show_n=False,base_line_width=2.0,show_quantilles=False,show_scatter=False,y_max=14)
    # show(p)
    
    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='L PD', style='violin',y_scaling=1,y_name=r'L proximal-distal $$\mu m$$',test_significance=False,
    #                                   y0=0,show_n=False,base_line_width=2.0,show_quantilles=False,y_max=600)
    # show(p)

    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='L AP', style='violin',y_scaling=1,y_name=r'L anterior-posterior $$\mu m$$',test_significance=False,
    #                                   y0=0,show_n=False,base_line_width=2.0,show_quantilles=False,y_max=600)
    # show(p)

    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='V / A', style='box',y_scaling=1,y_name=r'Mean Thickness $$\mu m$$',test_significance=False,
    #                                   y0=0,show_n=False,base_line_width=2.0,show_quantilles=False,show_scatter=False,y_max=50)
    # show(p)
    

    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    # p=add_data_to_plot_II(df,p,y_col='Surface Area',category='4850cut',y_scaling=1e-4,color='black',width=width)
    # p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='Surface Area',color='black',y_scaling=1e-4)
    # show(p)
    
    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$',test_significance=True,y0=0)
    # p=add_data_to_plot_II(df,p,y_col='Volume',category='4850cut',y_scaling=1e-6,color='black',width=width)
    # p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='Volume',color='black',y_scaling=1e-6)
    # show(p)

    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='V / A', style='violin',y_scaling=1,y_name=r'Mean Thickness $$\mu m$$',test_significance=True,y0=0)
    # p=add_data_to_plot_II(df,p,y_col='V / A',category='4850cut',y_scaling=1,color='black',width=width)
    # p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='V / A',color='black')
    # show(p)

    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='L PD', style='violin',y_scaling=1,y_name=r'L proximal-distal $$\mu m$$',test_significance=True,y0=0)
    # p=add_data_to_plot_II(df,p,y_col='L PD',category='4850cut',y_scaling=1,color='black',width=width)
    # p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='L PD',color='black')
    # show(p)

    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='L AP', style='violin',y_scaling=1,y_name=r'L anterior-posterior $$\mu m$$',test_significance=True,y0=0)
    # p=add_data_to_plot_II(df,p,y_col='L AP',category='4850cut',y_scaling=1,color='black',width=width)
    # p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='L AP',color='black')
    # show(p)

    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='L DV', style='violin',y_scaling=1,y_name=r'L dorso-ventral $$\mu m$$',test_significance=True,y0=0)
    # p=add_data_to_plot_II(df,p,y_col='L DV',category='4850cut',y_scaling=1,color='black',width=width)
    # p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='L DV',color='black')
    # show(p)

    
    # p=plot_scatter_II(df, x_col='L DV', y_col='V / A', mode='category',show_fit=True,show_div='Residual')
    # p=add_data_scatter_II(df,p,x_col='L DV', y_col='V / A',category='4850cut',color='black')
    # show(p)

    # p=plot_scatter_II(df, x_col='Surface Area', y_col='L AP * L PD', mode='category',show_fit=True,show_div='Residual')
    # p=add_data_scatter_II(df,p, x_col='Surface Area', y_col='L AP * L PD',category='4850cut',color='black')
    # show(p)

    # p=plot_scatter_II(df, x_col='Volume', y_col='Surface Area', mode='category',show_fit=True,show_div='Residual')
    # p=add_data_scatter_II(df,p,x_col='Volume', y_col='Surface Area',category='4850cut',color='black')
    # show(p)
    
    p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    p=add_data_to_plot_II(df,p,y_col='Surface Area',category='7230cut',y_scaling=1e-4,color='purple',width=width)
    p=add_lines_to_time_series(df,time_series_list_7230,p,y_col='Surface Area',color='purple',y_scaling=1e-4)
    show(p)
    p=add_data_to_plot_II(df,p,y_col='Surface Area',category='4850cut',y_scaling=1e-4,color='black',width=width)
    p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='Surface Area',color='black',y_scaling=1e-4)
    show(p)
    
    p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$',test_significance=True,y0=0)
    p=add_data_to_plot_II(df,p,y_col='Volume',category='7230cut',y_scaling=1e-6,color='purple',width=width)
    p=add_lines_to_time_series(df,time_series_list_7230,p,y_col='Volume',color='purple',y_scaling=1e-6)
    show(p)
    p=add_data_to_plot_II(df,p,y_col='Volume',category='4850cut',y_scaling=1e-6,color='black',width=width)
    p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='Volume',color='black',y_scaling=1e-6)
    show(p)

    p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='V / A', style='violin',y_scaling=1,y_name=r'Mean Thickness $$\mu m$$',test_significance=True,y0=0)
    p=add_data_to_plot_II(df,p,y_col='V / A',category='7230cut',y_scaling=1,color='purple',width=width)
    p=add_lines_to_time_series(df,time_series_list_7230,p,y_col='V / A',color='purple')
    show(p)
    p=add_data_to_plot_II(df,p,y_col='V / A',category='4850cut',y_scaling=1,color='black',width=width)
    p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='V / A',color='black')
    show(p)

    p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='L PD', style='violin',y_scaling=1,y_name=r'L proximal-distal $$\mu m$$',test_significance=True,y0=0)
    p=add_data_to_plot_II(df,p,y_col='L PD',category='7230cut',y_scaling=1,color='purple',width=width)
    p=add_lines_to_time_series(df,time_series_list_7230,p,y_col='L PD',color='purple')
    show(p)
    p=add_data_to_plot_II(df,p,y_col='L PD',category='4850cut',y_scaling=1,color='black',width=width)
    p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='L PD',color='black')
    show(p)

    p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='L AP', style='violin',y_scaling=1,y_name=r'L anterior-posterior $$\mu m$$',test_significance=True,y0=0)
    p=add_data_to_plot_II(df,p,y_col='L AP',category='7230cut',y_scaling=1,color='purple',width=width)
    p=add_lines_to_time_series(df,time_series_list_7230,p,y_col='L AP',color='purple')
    show(p)
    p=add_data_to_plot_II(df,p,y_col='L AP',category='4850cut',y_scaling=1,color='black',width=width)
    p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='L AP',color='black')
    show(p)

    p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='L DV', style='violin',y_scaling=1,y_name=r'L dorso-ventral $$\mu m$$',test_significance=True,y0=0)
    p=add_data_to_plot_II(df,p,y_col='L DV',category='7230cut',y_scaling=1,color='purple',width=width)
    p=add_lines_to_time_series(df,time_series_list_7230,p,y_col='L DV',color='purple')
    show(p)
    p=add_data_to_plot_II(df,p,y_col='L DV',category='4850cut',y_scaling=1,color='black',width=width)
    p=add_lines_to_time_series(df,time_series_list_4850,p,y_col='L DV',color='black')
    show(p)


import matplotlib.pyplot as plt
def plot_variance(df, quantity='Volume'):
    """
    Plots the mean with standard deviation error bars over time for each condition.
    Also plots the standard deviation with its error to analyze the spread of values.
    
    Parameters:
    df (pd.DataFrame): The dataset containing time in hpf, condition, and the quantity to analyze.
    quantity (str): The column name of the quantity to plot.
    """

    # Define colors for each condition
    condition_colors = {
        'Development': 'blue',
        'Regeneration': 'orange',
        '4850cut': 'black'
    }

    # Group data by condition and time
    grouped = df.groupby(['condition', 'time in hpf'])[quantity]

    # Compute mean, std, and std error
    stats = grouped.agg(['mean', 'std', 'count']).reset_index()
    stats['std_err_std'] = stats['std'] / np.sqrt(2 * (stats['count'] - 1))

    # Create figure with two subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 20), sharex=True)

    # Plot mean with std error
    for condition, color in condition_colors.items():
        subset = stats[stats['condition'] == condition]
        axs[0].errorbar(subset['time in hpf'], subset['mean'], yerr=subset['std'], 
                        fmt='o-', label=condition, color=color, capsize=5)
    
    axs[0].set_ylabel(f'Mean {quantity}')
    axs[0].set_title(f'Time Evolution of {quantity}')
    axs[0].legend()
    axs[0].grid(True)

    # Plot std with its error
    for condition, color in condition_colors.items():
        subset = stats[stats['condition'] == condition]
        axs[1].errorbar(subset['time in hpf'], subset['std'], yerr=subset['std_err_std'], 
                        fmt='s-', label=condition, color=color, capsize=5)
    
    axs[1].set_ylabel(f'Standard Deviation of {quantity}')
    axs[1].legend()
    axs[1].grid(True)

     # Plot std with its error
    for condition, color in condition_colors.items():
        subset = stats[stats['condition'] == condition]
        axs[2].errorbar(subset['time in hpf'], subset['std']/subset['mean'], yerr=subset['std_err_std']/subset['mean'], 
                        fmt='s-', label=condition, color=color, capsize=5)
    
    axs[2].set_xlabel('Time in hpf')
    axs[2].set_ylabel(f'Relative Standard Deviation of {quantity}')
    axs[2].legend()
    axs[2].grid(True)

    plt.show()

def plot_compare_CoV(df, quantity='Volume'):
    """
    Plots a bar chart comparing coefficient of variation (CoV = std/mean) between
    48 hpf and 144 hpf for each condition, with error bars.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns 'time in hpf', 'condition', and the target quantity.
    quantity (str): The column name of the quantity to analyze.
    """

    # Only use data from 48 and 144 hpf
    df_filtered = df[df['time in hpf'].isin([48, 144])]

    # Group by condition and time
    grouped = df_filtered.groupby(['condition', 'time in hpf'])[quantity]
    stats = grouped.agg(['mean', 'std', 'count']).reset_index()

    # Coefficient of Variation
    stats['cov'] = stats['std'] / stats['mean']

    # Error of CoV: derived from delta method
    stats['cov_err'] = stats['cov'] * np.sqrt(1/(2 * (stats['count'] - 1)) + (stats['std']**2) / (stats['mean']**2 * stats['count']))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.35
    timepoints = [48, 144]
    conditions = ['Development', 'Regeneration']

    x = np.arange(len(timepoints))
    for i, cond in enumerate(conditions):
        subset = stats[stats['condition'] == cond]
        cov_vals = subset.set_index('time in hpf').loc[timepoints, 'cov']
        cov_errs = subset.set_index('time in hpf').loc[timepoints, 'cov_err']
        ax.bar(x + i * width, cov_vals, width, yerr=cov_errs, capsize=5, label=cond)

    ax.set_xticks(x + width * (len(conditions) - 1) / 2)
    ax.set_xticklabels([f'{t} hpf' for t in timepoints])
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title(f'CoV of {quantity} at 48 and 144 hpf')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_all_variance():
    df=getData()
    
    
    print(df.columns)
    print(df.head())
    df['Volume']=df['Volume']*1e-6
    df['Surface Area']=df['Surface Area']*1e-4

    plot_compare_CoV(df, quantity='Surface Area')
    return


    plot_variance(df, quantity='Volume')
    plot_variance(df, quantity='Surface Area')
    plot_variance(df, quantity='V / A')



if __name__ == "__main__":
    #main()
    new_plots()

    #plot_all_variance()