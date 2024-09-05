import pandas as pd
import os
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, FactorRange
from bokeh.palettes import Viridis256
from bokeh.transform import linear_cmap

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

    return df

def plot_scatter(df, x_col, y_col, mode='category', tooltips=None):
    """
    Create a scatter plot using Bokeh with flexible axis assignment and hover tools.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data to plot.
    x_col : str
        The column name from `df` to use for the x-axis.
    y_col : str
        The column name from `df` to use for the y-axis.
    mode : str, optional (default='category')
        Determines how the plot is color-coded and shaped.
        'category' : Colors the plot based on the 'condition' column (Development or Regeneration).
        'time'     : Colors the plot based on 'time in hpf' column and shapes by the 'condition'.
    tooltips : list of tuples, optional (default=None)
        List of custom hover tooltips to show. If None, default hover tooltips are shown.
        Example: [("Column Name", "@column_name"), ...]

    Returns:
    --------
    None
    Displays an interactive scatter plot in the browser.
    """

    # Define color palettes and marker shapes
    colors = {'Development': 'blue', 'Regeneration': 'orange'}
    shapes = {'Development': 'circle', 'Regeneration': 'triangle'}
    
    # Create the figure
    p = figure(title=f"{x_col} vs {y_col}",
               x_axis_label=x_col,
               y_axis_label=y_col,
               tools="pan,wheel_zoom,box_zoom,reset,save")
    
    # Define default hover tooltips if not provided
    if tooltips is None:
        tooltips = [
            ("Mask Folder", "@{Mask Folder}"),
            (x_col, f"@{{{x_col}}}"),
            (y_col, f"@{{{y_col}}}"),
            ("Surface Area", "@{Surface Area}"),
            ("Integrated Thickness", "@{Integrated Thickness}"),
            ("L PD", "@{L PD}"),
            ("L AP", "@{L AP}"),
            ("Condition", "@{condition}"),
            ("Time (hpf)", "@{time in hpf}"),
            ("Experimentalist", "@{experimentalist}"),
            ("Genotype", "@{genotype}")
        ]
    
    hover = HoverTool(tooltips=tooltips)
    p.add_tools(hover)
    
    # Plot based on mode
    if mode == 'category':  # Color by category (Development/Regeneration)
        for condition, color in colors.items():
            shape = shapes[condition]
            subset = df[df['condition'] == condition]
            source_subset = ColumnDataSource(subset)
            
           # Use scatter() with marker shape and size
            p.scatter(x=x_col, y=y_col, source=source_subset,
                      size=10, color=color, marker=shape, legend_label=condition, alpha=0.6)
    
    
    elif mode == 'time':  # Color by time, shape by category
        # Time-based colormap
        time_mapper = linear_cmap(field_name='time in hpf', palette=Viridis256, low=df['time in hpf'].min(), high=df['time in hpf'].max())
        
        for condition, shape in shapes.items():
            subset = df[df['condition'] == condition]
            source_subset = ColumnDataSource(subset)
            
            # Use scatter() with marker shape and size, color mapped by time
            p.scatter(x=x_col, y=y_col, source=source_subset,
                      size=10, color=time_mapper, marker=shape, legend_label=condition, alpha=0.6)
    
    # Configure legend
    p.legend.title = 'Condition'
    p.legend.location = "top_left"
    
    # Show plot
    show(p)

def plot_single_timeseries(df, filter_col=None, filter_value=None, y_col=None, style='box', color='blue'):
    """
    Create a time-series plot (box plot or violin plot) with individual scatter points, based on filtered data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data to plot.
    filter_col : str, optional (default=None)
        The column to filter by (e.g., 'condition', 'genotype', or 'experimentalist'). If None, no filtering is applied.
    filter_value : str or int, optional (default=None)
        The specific value to filter by within `filter_col`. If None, no filtering is applied.
    y_col : str
        The column name from `df` to use for the y-axis (quantity of interest).
    style : str, optional (default='box')
        The plot style: 'box' for box plot or 'violin' for violin plot.
    color : str, optional (default='blue')
        The color for the scatter points and the plots.
    
    Returns:
    --------
    None
    Displays an interactive time-series plot (box or violin) with scatter points in the browser.
    """

    # Apply filtering if specified
    if filter_col and filter_value:
        df = df[df[filter_col] == filter_value]
    
    # Check if y_col is provided
    if y_col is None:
        raise ValueError("You must specify the column for the y-axis (y_col).")
    
    # Ensure 'time in hpf' exists and is integer
    df['time in hpf'] = df['time in hpf'].astype(int)

    # Group data by 'time in hpf'
    grouped = df.groupby('time in hpf')
    
    # Extract unique times and prepare the data for boxplot/violin plot
    times = list(grouped.groups.keys())
    values = [grouped.get_group(time)[y_col].values for time in times]
    
    # Create a ColumnDataSource for scatter plot
    scatter_source = ColumnDataSource(df)
    
    # Create the figure
    p = figure(title=f"Time-series ({style.capitalize()} Plot) of {y_col} by Time",
               x_axis_label="Time (hpf)",
               y_axis_label=y_col,
               tools="pan,wheel_zoom,box_zoom,reset,save",
               x_range=FactorRange(*[str(t) for t in times]))  # Force categorical x-axis
    
    # Add hover tool for scatter plot
    hover = HoverTool()
    hover.tooltips = [
        ("Time (hpf)", "@{time in hpf}"),
        (y_col, f"@{{{y_col}}}"),
        ("Condition", "@{condition}"),
        ("Genotype", "@{genotype}"),
        ("Experimentalist", "@{experimentalist}")
    ]
    p.add_tools(hover)
    
    # Scatter plot for individual points
    p.scatter(x='time in hpf', y=y_col, source=scatter_source,
              size=8, color=color, alpha=0.6, legend_label='Data Points')
    
    # Box or violin plot
    if style == 'box':
        # Add a box plot
        for i, (time, val) in enumerate(zip(times, values)):
            # Calculate the quartiles and whiskers
            q1, q2, q3 = np.percentile(val, [25, 50, 75])
            iqr = q3 - q1
            lower_bound = max(min(val), q1 - 1.5 * iqr)
            upper_bound = min(max(val), q3 + 1.5 * iqr)
            
            # Draw the boxes and whiskers
            p.vbar(x=str(time), width=0.5, bottom=q1, top=q3, color=color, alpha=0.4)
            p.segment(x0=[str(time)], y0=[lower_bound], x1=[str(time)], y1=[upper_bound], color='black')
            p.circle(x=[str(time)], y=[q2], size=10, color='black', fill_color='white')
    
    elif style == 'violin':
        # Draw a violin plot (mirror density plot)
        from scipy.stats import gaussian_kde
        
        for i, (time, val) in enumerate(zip(times, values)):
            kde = gaussian_kde(val, bw_method=0.3)  # Gaussian KDE for smoothing
            x = np.linspace(min(val), max(val), 100)
            kde_values = kde(x)
            
            # Scale and mirror the violin shape
            p.patch([str(time)] * len(x) + [str(time)] * len(x),
                    list(x) + list(x[::-1]), alpha=0.3, color=color,
                    line_color=color)
    
    # Configure the legend
    p.legend.location = "top_left"
    
    # Show the plot
    show(p)

def main():
    df=getData()
    print(df)

    #Checks
    # plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='category')
    # plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='time')

    # plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='category')
    # plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='time')

    # plot_scatter(df, x_col='V / A', y_col='Int_dA_d / A', mode='category')
    # plot_scatter(df, x_col='V / A', y_col='Int_dA_d / A', mode='time')


    plot_single_timeseries(df, filter_col='condition', filter_value='Regeneration', y_col='Volume', style='box', color='orange')


if __name__ == "__main__":
    main()