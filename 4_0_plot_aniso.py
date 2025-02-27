import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from bokeh.layouts import row

from plotHelper import plot_flexible_timeseries
from bokeh.io import show

from config import *



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

    print(df.columns)

    return df

def fit_and_plot_sklearn(cleaned_df,p):
    x = cleaned_df['L PD'].values.reshape(-1, 1)
    y = cleaned_df['L AP'].values

    # Fit 1: Linear through origin
    model_origin = LinearRegression(fit_intercept=False)
    model_origin.fit(x, y)
    print(f"Fit 1 (Linear through origin): Coefficient = {model_origin.coef_[0]}")
    

    # Fit 2: Linear with offset
    model_offset = LinearRegression(fit_intercept=True)
    model_offset.fit(x, y)
    print(f"Fit 2 (Linear with offset): Coefficient = {model_offset.coef_[0]}, Intercept = {model_offset.intercept_}")


    # Fit 3: Logarithmic linear fit
    log_x = np.log(x).reshape(-1, 1)
    log_y = np.log(y)
    model_log = LinearRegression(fit_intercept=True)
    model_log.fit(log_x, log_y)
    print(f"Fit 3 (Logarithmic linear fit): Coefficient = {model_log.coef_[0]}, Intercept = {model_log.intercept_}")

    # Generate x range for fit lines
    x_fit = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    y_fit_origin_line = model_origin.predict(x_fit)
    y_fit_offset_line = model_offset.predict(x_fit)
    y_fit_log_line = np.exp(model_log.predict(np.log(x_fit)))

    # Plot the results
    
    p.line(x_fit.flatten(), y_fit_origin_line, legend_label="Linear Fit (Origin)", line_color="black", line_width=2, line_dash="dotted")
    p.line(x_fit.flatten(), y_fit_offset_line, legend_label="Linear Fit (Offset)", line_color="black", line_width=2, line_dash="dashed")
    p.line(x_fit.flatten(), y_fit_log_line, legend_label="Log Fit", line_color="black", line_width=2, line_dash="solid")

    # Add legends and formatting
    p.legend.location = "top_left"
    return p

def plot():
    cleaned_df = getData()

    print(cleaned_df.head())


    # Define the color and marker dictionaries
    color_dict = {'Regeneration': 'orange',
                 'Development': 'blue', 
                 '72FF_cut':'black',
                 '4850cut':'grey',
                 '48FF_cut':'grey'
                 }
    marker_dict = {'Development': 'circle', 'Regeneration': 'triangle','72FF_cut':'square','4850cut':'circle','48FF_cut':'circle'}

    _, p, _ = plot_flexible_timeseries(
        df=cleaned_df,
        time_col='time in hpf',
        value_A_col='L PD',
        value_B_col='L AP',
        category_col='condition',
        color_dict=color_dict,
        marker_dict=marker_dict,
        color_by='time', 
        show_candle=True  ,
        A_axis_label=r'$$L_{PD}$$ in $$ \mu m$$',
        B_axis_label=r'$$L_{AP}$$ in $$\mu m$$'
    )

    fit_and_plot_sklearn(cleaned_df,p)

    from bokeh.models import LogScale
    p.x_scale=LogScale()
    p.y_scale=LogScale()

    show(p)
    
    _, p, _ = plot_flexible_timeseries(
        df=cleaned_df,
        time_col='time in hpf',
        value_A_col='L PD',
        value_B_col='L AP',
        category_col='condition',
        color_dict=color_dict,
        marker_dict=marker_dict,
        color_by='category', 
        show_candle=True  ,
        A_axis_label=r'$$L_{PD}$$ in $$ \mu m$$',
        B_axis_label=r'$$L_{AP}$$ in $$\mu m$$'
    )

    fit_and_plot_sklearn(cleaned_df,p)

    from bokeh.models import LogScale
    p.x_scale=LogScale()
    p.y_scale=LogScale()

    show(p)

    _, p, _ = plot_flexible_timeseries(
        df=cleaned_df,
        time_col='time in hpf',
        value_A_col='L PD',
        value_B_col='L AP',
        category_col='condition',
        color_dict=color_dict,
        marker_dict=marker_dict,
        color_by='time', 
        show_candle=True  ,
        A_axis_label=r'$$L_{PD}$$ in $$ \mu m$$',
        B_axis_label=r'$$L_{AP}$$ in $$\mu m$$'
    )

    fit_and_plot_sklearn(cleaned_df,p)


    show(p)

    _, p, _ = plot_flexible_timeseries(
        df=cleaned_df,
        time_col='time in hpf',
        value_A_col='L PD',
        value_B_col='L AP',
        category_col='condition',
        color_dict=color_dict,
        marker_dict=marker_dict,
        color_by='category', 
        show_candle=True  ,
        A_axis_label=r'$$L_{PD}$$ in $$ \mu m$$',
        B_axis_label=r'$$L_{AP}$$ in $$\mu m$$'
    )

    fit_and_plot_sklearn(cleaned_df,p)


    show(p)

def plot_full():
    cleaned_df = getData()

    print(cleaned_df.columns)
    cleaned_df=cleaned_df[cleaned_df["condition"].isin(["Development", "Regeneration"])]

    # Define the color and marker dictionaries
    color_dict = {'Regeneration': 'orange',
                 'Development': 'blue', 
                #  '72FF_cut':'black',
                #  '48FF_cut':'grey'
                 }
    marker_dict = {'Development': 'circle', 'Regeneration': 'triangle'}

    p_left, p_middle, p_right = plot_flexible_timeseries(
        df=cleaned_df,
        time_col='time in hpf',
        value_A_col='L PD',
        value_B_col='L AP',
        category_col='condition',
        color_dict=color_dict,
        marker_dict=marker_dict,
        color_by='time', 
        show_candle=True  ,
        A_axis_label=r'$$L_{PD}$$ in $$ \mu m$$',
        B_axis_label=r'$$L_{AP}$$ in $$\mu m$$'
    )

   
    show(row(p_left, p_middle, p_right))
   
    
    p_left, p_middle, p_right = plot_flexible_timeseries(
        df=cleaned_df,
        time_col='time in hpf',
        value_A_col='L PD',
        value_B_col='L AP',
        category_col='condition',
        color_dict=color_dict,
        marker_dict=marker_dict,
        color_by='category', 
        show_candle=True  ,
        A_axis_label=r'$$L_{PD}$$ in $$ \mu m$$',
        B_axis_label=r'$$L_{AP}$$ in $$\mu m$$'
    )

    show(row(p_left, p_middle, p_right))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_matplotlib():
    cleaned_df = getData()

    # Define the color and marker dictionaries
    color_dict = {'Regeneration': 'orange', 'Development': 'blue', '4850cut':'black'}  # Matplotlib colors
    marker_dict = {'Development': 'o', 'Regeneration': '^', '4850cut': 'x'}  # Matplotlib markers

    # Filter the DataFrame to keep only valid conditions
    cleaned_df = cleaned_df[cleaned_df['condition'].isin(color_dict.keys())]

    # Extract relevant columns and apply log transformation
    log_L_PD = np.log(cleaned_df['L PD'])
    log_L_AP = np.log(cleaned_df['L AP'])

    # Fit a power-law model (linear fit in log-log space)
    slope, intercept, r_value, _, _ = linregress(log_L_PD, log_L_AP)
    anisotropy = slope  # Slope is the anisotropy measure

    # Generate fit line for plotting
    log_L_PD_fit = np.linspace(log_L_PD.min(), log_L_PD.max(), 100)
    log_L_AP_fit = intercept + slope * log_L_PD_fit  # Power-law best fit

    # Fit isotropic line with forced slope=1
    mean_intercept = np.mean(log_L_AP - log_L_PD)  # Compute mean vertical offset
    log_L_AP_iso = log_L_PD_fit + mean_intercept  # Isotropic reference line

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter plot of the data
    for condition in cleaned_df['condition'].unique():
        subset = cleaned_df[cleaned_df['condition'] == condition]
        ax.scatter(np.log(subset['L PD']), np.log(subset['L AP']),
                   label=condition, color=color_dict[condition], marker=marker_dict[condition])

    # Plot power-law fit
    ax.plot(log_L_PD_fit, log_L_AP_fit, 'k-', label=f'Power-law fit (slope={slope:.2f})')

    # Plot isotropic reference line with forced slope=1
    ax.plot(log_L_PD_fit, log_L_AP_iso, 'k--', label='Isotropic fit (slope=1)')

    # Labels and legend
    ax.set_xlabel('log L PD')
    ax.set_ylabel('log L AP')
    ax.set_title('Log-Log Plot of L PD vs L AP')
    ax.legend()
    
    fig.savefig("/home/max/Downloads/log_log_plot.png", dpi=300, bbox_inches='tight')
    fig.savefig("/home/max/Downloads/log_log_plot.pdf", bbox_inches='tight')
    fig.savefig("/home/max/Downloads/log_log_plot.svg", bbox_inches='tight')

    plt.show()

    # Print anisotropy estimate
    print(f"Estimated Anisotropy (slope of power law fit): {anisotropy:.2f}")


if __name__ == "__main__":
    #plot_full()
    #plot()

    plot_matplotlib()