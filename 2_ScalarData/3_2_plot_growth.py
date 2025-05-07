import pandas as pd
import os
import numpy as np

from plotHelper.plotBokehHelper_old import plot_scatter_corner
from bokeh.io import show

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

from plotHelper.bokeh_timeseries_plot import plot_double_timeseries, add_data_to_plot, add_lines_to_time_series, plot_explanation

    
def getData():
    df_file_path = os.path.join(scalar_path,'scalarGrowthData.csv')

    # Load the DataFrame from the HDF5 file
    df = pd.read_csv(df_file_path,sep=';')

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


def corner_plots():
    df=getData()
    
    print(df.columns)

    #things with ED
    df = df.dropna()

    color_dict = {'Regeneration': 'orange',
                 'Development': 'blue', 
                 }
    marker_dict = {'Development': 'circle', 'Regeneration': 'triangle', 'Smoc12': 'square'}
    corner_plot = plot_scatter_corner(df=df, parameters=['Volume','Surface Area','V / A', 'ED Mask Volume','N ED cells'], color_col='condition',color_dict=color_dict,marker_col='condition',marker_dict=marker_dict)
    show(corner_plot)
    corner_plot = plot_scatter_corner(df=df, parameters=['Volume','Surface Area','V / A', 'ED Mask Volume','N ED cells'], color_col='time in hpf',color_dict=color_dict,marker_col='condition',marker_dict=marker_dict)
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

def plot_time_series(plot_lines=True, include_7230cut=True, include_4850cut=True):
    df = getData()
    print(df.columns)
    print(df.head())

    condition_series = {}
    if include_4850cut:
        filtered_df = df[df["condition"] == "4850cut"]
        condition_series['4850cut'] = group_time_series(filtered_df)
    if include_7230cut:
        filtered_df = df[df["condition"] == "7230cut"]
        condition_series['7230cut'] = group_time_series(filtered_df)

    plots = [
        ('Surface Area', r'Area $$(100 \mu m)^2$$', 1e-4),
        ('Volume', r'Volume $$(100 \mu m)^3$$', 1e-6),
        ('V / A', r'Mean Thickness $$\mu m$$', 1),
        ('L PD', r'L proximal-distal $$\mu m$$', 1),
        ('L AP', r'L anterior-posterior $$\mu m$$', 1),
        ('L DV', r'L dorso-ventral $$\mu m$$', 1),
        ('ED Mask Volume', r'Volume ED $$(100 \mu m)^3$$', 1e-6),
        ('N ED cells', r'$$N_{ED}$$', 1),
        ('Surface Area ED', r'Area ED $$(100 \mu m)^2$$', 1e-4)
    ]

    colors = {'7230cut': 'purple', '4850cut': 'black'}
    categories = {'Development': 'blue', 'Regeneration': 'orange'}

    for y_col, y_label, y_scale in plots:
        df_cleaned = df.dropna(subset=[y_col])

        p, width = plot_double_timeseries(
            df_cleaned,
            categories=categories,
            y_col=y_col,
            style='violin',
            y_scaling=y_scale,
            y_name=y_label,
            test_significance=True,
            y0=0
        )

        for cond in ['7230cut', '4850cut']:
            if cond in condition_series:
                p = add_data_to_plot(
                    df_cleaned,
                    p,
                    y_col=y_col,
                    category=cond,
                    y_scaling=y_scale,
                    color=colors[cond],
                    width=width
                )
                if plot_lines:
                    p = add_lines_to_time_series(
                        df_cleaned,
                        condition_series[cond],
                        p,
                        y_col=y_col,
                        color=colors[cond],
                        y_scaling=y_scale
                    )

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
    plot_compare_CoV(df, quantity='Volume')
    plot_compare_CoV(df, quantity='L PD')
    plot_compare_CoV(df, quantity='L AP')
    plot_compare_CoV(df, quantity='L DV')
    plot_compare_CoV(df, quantity='V / A')
    return


    plot_variance(df, quantity='Volume')
    plot_variance(df, quantity='Surface Area')
    plot_variance(df, quantity='V / A')

def plot_scatter_data():
    from plotHelper.bokeh_scatter_plot import plot_scatter
    from bokeh.io import show

    df = getData()
    print(df.columns)

    colors = {
        'Regeneration': 'orange',
        'Development': 'blue',
        '4850cut': 'black',
        '7230cut': 'purple'
    }

    shapes = {
        'Regeneration': 'triangle',
        'Development': 'circle'
    }

    show(plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='category', show_fit=True, show_div='Residual', colors=colors, shapes=shapes))
    show(plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='time', show_fit=True, show_div='Residual', colors=colors, shapes=shapes))

    show(plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='category', show_fit=True, show_div='Residual', colors=colors, shapes=shapes))
    show(plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='time', show_fit=True, show_div='Residual', colors=colors, shapes=shapes))

    show(plot_scatter(df, x_col='L AP', y_col='L PD', mode='category', show_fit=True, show_div='Residual', colors=colors, shapes=shapes))
    show(plot_scatter(df, x_col='L AP', y_col='L PD', mode='time', show_fit=True, show_div='Residual', colors=colors, shapes=shapes))

    show(plot_scatter(df, x_col='log L PD', y_col='log L AP', x_name=r'$$log(L_{PD})$$', y_name=r'$$log(L_{AP})$$', mode='category', show_fit=True, show_div='Residual', show_slope_1=True, colors=colors, shapes=shapes))
    show(plot_scatter(df, x_col='log L PD', y_col='log L AP', x_name=r'$$log(L_{PD})$$', y_name=r'$$log(L_{AP})$$', mode='time', show_fit=True, show_div='Residual', colors=colors, shapes=shapes))

if __name__ == "__main__":
    # show(plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='whiskers', title="Whiskers Explanation"))
    # show(plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='box', title="Box Plot Explanation"))
    # show(plot_explanation(lower_bound=10, q1=20, q2=30, q3=40, upper_bound=50, plot_type='violin', title="Violin Plot Explanation"))
    #corner_plots()
    plot_time_series()
    plot_all_variance()
    plot_scatter_data()