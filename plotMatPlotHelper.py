import matplotlib.pyplot as plt
import seaborn as sns

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
        df = df.loc[df[filter_col] == filter_value].copy()

    # Check if y_col is provided
    if y_col is None:
        raise ValueError("You must specify the column for the y-axis (y_col).")
    
    # Ensure 'time in hpf' exists and is treated as a categorical variable
    df['time in hpf'] = df['time in hpf'].astype(int)

    # Set the plot size
    plt.figure(figsize=(10, 6))

    # Create the boxplot or violin plot depending on the style parameter
    if style == 'box':
        # Create a boxplot using seaborn
        sns.boxplot(x='time in hpf', y=y_col, data=df, color=color, width=0.6)
    elif style == 'violin':
        # Create a violin plot using seaborn
        sns.violinplot(x='time in hpf', y=y_col, data=df, color=color, width=0.8, inner=None)

    # Add scatter points (raw data points) over the plot
    sns.stripplot(x='time in hpf', y=y_col, data=df, color='black', size=6, jitter=True, alpha=0.6)

    # Add titles and labels
    plt.title(f"Time-series ({style.capitalize()} Plot) of {y_col} by Time")
    plt.xlabel("Time (hpf)")
    plt.ylabel(y_col)

    # Show the plot
    plt.show()

def plot_double_timeseries(df, y_col=None, style='box'):
    """
    Create a time-series plot (box plot or violin plot) with individual scatter points, based on filtered data.
    The plot will be split in the middle: 
    - Left: 'Development' in blue
    - Right: 'Regeneration' in orange
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the data to plot.
    y_col : str
        The column name from `df` to use for the y-axis (quantity of interest).
    style : str, optional (default='box')
        The plot style: 'box' for box plot or 'violin' for violin plot.
    
    Returns:
    --------
    None
    Displays an interactive time-series plot (split box or violin) with scatter points in the browser.
    """
    
    # Check if y_col is provided
    if y_col is None:
        raise ValueError("You must specify the column for the y-axis (y_col).")
    
    # Ensure 'time in hpf' exists and is treated as a categorical variable
    df['time in hpf'] = df['time in hpf'].astype(int)

    # Set the plot size
    plt.figure(figsize=(10, 6))
    
    if style == 'box':
        # Create boxplots split by condition using hue
        sns.boxplot(x='time in hpf', y=y_col, data=df, hue='condition', palette={'Development': 'blue', 'Regeneration': 'orange'}, width=0.6, dodge=True)
    
    elif style == 'violin':
        # Create violin plots split by condition using hue
        sns.violinplot(x='time in hpf', y=y_col, data=df, hue='condition', palette={'Development': 'blue', 'Regeneration': 'orange'}, width=0.8, inner=None, dodge=True)
    
    # Overlay scatter points for both conditions
    sns.stripplot(x='time in hpf', y=y_col, data=df, hue='condition', dodge=True, color='black', size=6, jitter=True, alpha=0.6)
    
    # Add titles and labels
    plt.title(f"Double Time-series ({style.capitalize()} Plot) of {y_col} by Time")
    plt.xlabel("Time (hpf)")
    plt.ylabel(y_col)
    
    # Show the plot
    plt.show()