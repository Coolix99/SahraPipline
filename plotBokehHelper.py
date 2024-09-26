import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, FactorRange, Div, Label
from bokeh.palettes import Viridis256
from bokeh.transform import linear_cmap
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from bokeh.layouts import column, row
from scipy import stats
from bokeh.layouts import gridplot
from scipy.stats import mannwhitneyu


def plot_scatter(df, x_col, y_col, x_name=None,y_name=None,mode='category', tooltips=None, show_fit=False, show_div='Residual'):
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
    show_fit : bool, optional (default=False)
        Whether to show the linear fit line.
    show_div : str, optional (default='Residual')
        Whether to show the deviation scatter plot. Options are 'Residual' or 'PCA'.

    Returns:
    --------
    None
    Displays an interactive scatter plot in the browser.
    """

    if y_name is None:
        y_name = y_col
    if x_name is None:
        x_name = x_col
    

    # Define color palettes and marker shapes
    colors = {'Development': 'blue', 'Regeneration': 'orange'}
    shapes = {'Development': 'circle', 'Regeneration': 'triangle'}
    
    # Create the figure
    p = figure(title=f"{x_name} vs {y_name}",
               x_axis_label=x_name,
               y_axis_label=y_name,
               width=700, height=400,
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
    
    
    # Plot based on mode
    if mode == 'category':  # Color by category (Development/Regeneration)
        for condition, color in colors.items():
            shape = shapes[condition]
            subset = df[df['condition'] == condition]
            source_subset = ColumnDataSource(subset)
            
           # Use scatter() with marker shape and size
            scatter_data=p.scatter(x=x_col, y=y_col, source=source_subset,
                      size=10, color=color, marker=shape, legend_label=condition, alpha=0.6)
    
    
    elif mode == 'time':  # Color by time, shape by category
        # Time-based colormap
        time_mapper = linear_cmap(field_name='time in hpf', palette=Viridis256, low=df['time in hpf'].min(), high=df['time in hpf'].max())
        
        for condition, shape in shapes.items():
            subset = df[df['condition'] == condition]
            source_subset = ColumnDataSource(subset)
            
            # Use scatter() with marker shape and size, color mapped by time
            scatter_data=p.scatter(x=x_col, y=y_col, source=source_subset,
                      size=10, color=time_mapper, marker=shape, legend_label=condition, alpha=0.6)
    
    hover = HoverTool(renderers=[scatter_data])
    hover.tooltips = tooltips
    p.add_tools(hover)

    # Configure legend
    p.legend.title = 'Condition'
    p.legend.location = "top_left"

    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.legend.label_text_font_size = "12pt"
    p.legend.title_text_font_size = "14pt"

    if show_fit or show_div:
        # Get x and y data
        x_data = df[x_col]
        y_data = df[y_col]
        times = df['time in hpf']
        # Perform linear regression (fit)
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        fit_line = slope * x_data + intercept
        print(x_col, y_col,'slope',slope, 'intercept',intercept,'std_err',std_err,'r_value',r_value)
        # Define the direction vector (unit vector) along the line
        n = np.array([1, slope]) / np.sqrt(1 + slope**2)  # Normalize to make it a unit vector
        o = np.array([slope, -1]) / np.sqrt(1 + slope**2)  # Normalize to make it a unit vector, orthogonal to n
        
        # Point a on the line (we take the intercept point, where x = 0)
        a = np.array([0, intercept])

        # Plot linear fit line
        if show_fit:
            p.line(x_data, fit_line, line_width=2, color='black', legend_label='Linear Fit')

        # Plot deviation if required
        if show_div=='PCA':
            deviation_fig = figure(title=f"Deviation: {x_name} vs {y_name}",
                                   x_axis_label='Orthogonal Position on Fit Line',
                                   y_axis_label='Distance from Line',
                                   tools="pan,wheel_zoom,box_zoom,reset,save",
                                   width=700, height=400)
            
            # Initialize the array to store the orthogonal distances and positions
            orthogonal_distances = []
            positions = []

            # Prepare lists for colors and shapes based on mode
            colors_list = []
            shapes_list = []
            
            # Create a new DataFrame to hold the deviation data
            deviation_df = df.copy()

            for i in range(len(x_data)):
                # Point p (x_i, y_i)
                p_point = np.array([x_data[i], y_data[i]])
                
                # Calculate the orthogonal distance using the vector formula
                a_p = a - p_point
                projection = np.dot(a_p, n) * n  # Project a_p onto the line
                orthogonal_vector = a_p - projection  # Perpendicular vector
                orthogonal_distance = np.dot(orthogonal_vector, o)
                
                orthogonal_distances.append(orthogonal_distance)
                positions.append(-np.dot(a_p, n))

                # Assign the same color and shape as the first plot based on mode
                condition = df['condition'].iloc[i]
                colors_list.append(colors[condition])
                shapes_list.append(shapes[condition])
           

            # Add the deviation data to the deviation_df
            deviation_df['time in hpf'] = times
            deviation_df['position'] = positions
            deviation_df['orthogonal_distance'] = orthogonal_distances
            deviation_df['colors'] = colors_list
            deviation_df['shapes'] = shapes_list

            # Create a new ColumnDataSource for the deviation plot
            deviation_source = ColumnDataSource(deviation_df)

            # Scatter plot for orthogonal distances with matching color and shapes
            if mode == 'category':
                deviation_scatter = deviation_fig.scatter(x='position', y='orthogonal_distance',
                                                        source=deviation_source, size=8,
                                                        color='colors', marker='shapes', alpha=0.6)
            elif mode == 'time':
                deviation_scatter = deviation_fig.scatter(x='position', y='orthogonal_distance',
                                                        source=deviation_source, size=8,
                                                        color=time_mapper, marker='shapes', alpha=0.6)

        
        elif show_div=='Residual':
            residuals = y_data - fit_line
            df['Residual'] = residuals

            # Create a new figure for residuals
            deviation_fig = figure(title=f"Residuals: {x_name} vs {y_name}",
                                  x_axis_label=x_name,
                                  y_axis_label='Residuals',
                                  tools="pan,wheel_zoom,box_zoom,reset,save",
                                  width=700, height=200)

            # Plot residuals based on mode
            if mode == 'category':  # Color by category (Development/Regeneration)
                for condition, color in colors.items():
                    shape = shapes[condition]
                    subset = df[df['condition'] == condition]
                    source_subset = ColumnDataSource(subset)
                    
                # Use scatter() with marker shape and size
                    deviation_scatter=deviation_fig.scatter(x=x_col, y='Residual', source=source_subset,
                            size=10, color=color, marker=shape, alpha=0.6)
            
            elif mode == 'time':  # Color by time, shape by category
                for condition, shape in shapes.items():
                    subset = df[df['condition'] == condition]
                    source_subset = ColumnDataSource(subset)
                    
                    # Use scatter() with marker shape and size, color mapped by time
                    deviation_scatter=deviation_fig.scatter(x=x_col, y='Residual', source=source_subset,
                            size=10, color=time_mapper, marker=shape, alpha=0.6)
                   
            deviation_fig.line(x=[x_data.min(), x_data.max()], y=[0, 0], color="black", line_width=2)  

        hover_deviation = HoverTool(renderers=[deviation_scatter])
        hover_deviation.tooltips = tooltips
        deviation_fig.add_tools(hover_deviation)

        deviation_fig.xaxis.axis_label_text_font_size = "14pt"
        deviation_fig.yaxis.axis_label_text_font_size = "14pt"
        deviation_fig.xaxis.major_label_text_font_size = "12pt"
        deviation_fig.yaxis.major_label_text_font_size = "12pt"
        
        layout = column(p, deviation_fig)
        show(layout)
        return


    # Show plot
    show(p)


def plot_single_timeseries(df, filter_col=None, filter_value=None, y_col=None, style='box', color='blue', width=None):
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
    width : float, optional (defualt=None)
        Scaling of the width of the bar or violin. If none a guess will be used
    
    Returns:
    --------
    None
    Displays an interactive time-series plot (box or violin) with scatter points in the browser.
    """

    # Apply filtering if specified
    if filter_col and filter_value:
        df = df.loc[df[filter_col] == filter_value].copy()  # Use .loc to avoid setting on copy of a slice warning
    
    # Check if y_col is provided
    if y_col is None:
        raise ValueError("You must specify the column for the y-axis (y_col).")

    # Ensure 'time in hpf' exists and is integer (numerical values)
    df['time in hpf'] = df['time in hpf'].astype(int)

    # Group data by 'time in hpf'
    grouped = df.groupby('time in hpf')
    
    # Extract unique times and prepare the data for boxplot/violin plot
    times = sorted(list(grouped.groups.keys()))  # Sorted list of time points
    values = [grouped.get_group(time)[y_col].values for time in times]
    
    if width is None:
        print(times)
        differences = [times[i+1] - times[i] for i in range(len(times) - 1)]
        smallest_interval = min(differences)
        width = smallest_interval*0.8

    # Create a ColumnDataSource for scatter plot
    scatter_source = ColumnDataSource(df)
    
    # Create the figure with numerical x-axis
    p = figure(title=f"Time-series ({style.capitalize()} Plot) of {y_col} by Time",
               x_axis_label="Time (hpf)",
               y_axis_label=y_col,
               tools="pan,wheel_zoom,box_zoom,reset,save")
    
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
            lower_bound,q1, q2, q3,upper_bound = np.percentile(val, [10,25, 50, 75,90])

            # the box
            p.vbar(x=time, width=width, bottom=q1, top=q3, color=color, alpha=0.4)

            
    elif style == 'violin':
        # Draw a violin plot (mirror density plot)
        for i, (time, val) in enumerate(zip(times, values)):
            # Gaussian KDE for smoothing
            kde = gaussian_kde(val, bw_method=0.3)
            
            # Define a range of y-values for the KDE (vertical axis)
            x = np.linspace(min(val)*0.8, max(val)*1.2, 100)
            
            # Get the density values for the KDE
            kde_values = kde(x)
            
            # Normalize the kde_values to fit the desired width in screen units
            kde_values = kde_values / kde_values.max() *width*0.5 # Normalize the density
            
            p.patch(np.concatenate([time - kde_values, (time + kde_values)[::-1]]),  # Left and right side of the violin
                np.concatenate([x, x[::-1]]),  # Mirror the vertical KDE on both sides
                alpha=0.3, color=color, line_color=color)
    
    #the lines
    for i, (time, val) in enumerate(zip(times, values)):
        # Calculate the quartiles and whiskers
        lower_bound,q1, q2, q3,upper_bound = np.percentile(val, [10,25, 50, 75,90])
        #the bounds
        p.segment(x0=[time], y0=[lower_bound], x1=[time], y1=[upper_bound], color='black',line_width=2)
        p.line(x=[time - width/3, time + width/3], y=[lower_bound, lower_bound], line_width=2, line_color='black', 
            line_dash='solid', line_cap='round', line_join='round')
        p.line(x=[time - width/3, time + width/3], y=[upper_bound, upper_bound], line_width=2, line_color='black', 
            line_dash='solid', line_cap='round', line_join='round')

        #the mean
        p.scatter(x=[time], y=[q2], size=10, color='black', marker='x', fill_color='white')  # Use scatter for median
        p.line(x=[time - width/2, time + width/2], y=[q2, q2], line_width=3, line_color='black', 
            line_dash='solid', line_cap='round', line_join='round')

    # Configure the legend
    p.legend.location = "top_left"
    
    # Show the plot
    show(p)

def plot_double_timeseries(df, y_col=None, style='box',y_scaling=1.0,y_name=None,test_significance=True,show_n=True,y0=None,fit_results=None,tight_style=True):
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
    Displays an interactive time-series plot (box or violin) with scatter points in the browser.
    """
    
    if y_col is None:
        raise ValueError("You must specify the column for the y-axis (y_col).")
    
    if y_name is None:
        y_name=y_col
    
    df=df.copy()
    df[y_col]=df[y_col]*y_scaling

    # Filter data for 'Development' and 'Regeneration'
    df_dev = df[df['condition'] == 'Development'].copy()
    df_reg = df[df['condition'] == 'Regeneration'].copy()

    # Ensure 'time in hpf' exists and is integer
    df_dev['time in hpf'] = df_dev['time in hpf'].astype(int)
    df_reg['time in hpf'] = df_reg['time in hpf'].astype(int)

    # Group data by 'time in hpf' for both conditions
    grouped_dev = df_dev.groupby('time in hpf')
    grouped_reg = df_reg.groupby('time in hpf')

    # Extract unique times and prepare data
    times_dev = sorted(list(grouped_dev.groups.keys()))
    values_dev = [grouped_dev.get_group(time)[y_col].values for time in times_dev]
    times_reg = sorted(list(grouped_reg.groups.keys()))
    values_reg = [grouped_reg.get_group(time)[y_col].values for time in times_reg]

    # Define plot width
    all_times = all_times = sorted(set(times_dev).union(set(times_reg)))
    width = 0.8 * min(np.diff(all_times)) if len(all_times) > 1 else 1
    
    # Apply the shift for scatter points
    df_dev['shifted_time'] = df_dev['time in hpf'] - width * 0.0  # Shift dev points to the left
    df_reg['shifted_time'] = df_reg['time in hpf'] + width * 0.0  # Shift reg points to the right

    # Create a ColumnDataSource for scatter plot
    scatter_source_dev = ColumnDataSource(df_dev)
    scatter_source_reg = ColumnDataSource(df_reg)

    max_y_value = max(df[y_col].max(), df[y_col].max()) * 1.2

    # Create figure with a shared numerical x-axis
    if y0 is None:
        # Let Bokeh automatically decide the y-range
        p = figure(title=f"Double Time-series ({style.capitalize()} Plot) of {y_name} by Time",
                   x_axis_label="Time (hpf)",
                   y_axis_label=y_name,
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=1000, height=600)
    
    else:
        # Set the y-range manually if y0 is specified
        p = figure(title=f"Double Time-series ({style.capitalize()} Plot) of {y_name} by Time",
                   x_axis_label="Time (hpf)",
                   y_axis_label=y_name,
                   y_range=(y0, max_y_value),  # Set both the lower and upper bounds
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=1000, height=600)

    # Scatter plot for individual points (Development)
    scatter_dev = p.scatter(x='shifted_time', y=y_col, source=scatter_source_dev,
              size=8, color='blue', alpha=0.6, legend_label='Development')

    # Scatter plot for individual points (Regeneration)
    scatter_reg = p.scatter(x='shifted_time', y=y_col, source=scatter_source_reg,
              size=8, color='orange', alpha=0.6, legend_label='Regeneration')

    if fit_results  is not None:
        for condition, color in zip(['Development', 'Regeneration'], ['blue', 'orange']):
            # Fit results for the given condition
            t_values = fit_results[f't_values']  # Theoretical fit values
            fit_noisy_vals = fit_results[f'{condition}']  # Noisy fit values
            
            # Compute mean and percentiles for the fits
            fit_mean = np.mean(fit_noisy_vals, axis=0)
            fit_25, fit_75 = np.percentile(fit_noisy_vals, [25, 75], axis=0)
            fit_10, fit_90 = np.percentile(fit_noisy_vals, [10, 90], axis=0)

            # Plot mean as a solid line
            p.line(t_values, fit_mean, color=color, line_width=2, legend_label=f'{condition} Fit')

            # Plot shaded regions for 25/75 and 10/90 percentiles
            p.varea(x=t_values, y1=fit_25, y2=fit_75, color=color, alpha=0.2)
            p.varea(x=t_values, y1=fit_10, y2=fit_90, color=color, alpha=0.1)

    if test_significance:
        for time in all_times:
            if time in grouped_dev.groups.keys() and time in grouped_reg.groups.keys():
                dev_vals = grouped_dev.get_group(time)[y_col].values
                reg_vals = grouped_reg.get_group(time)[y_col].values

                # Perform t-test
                #t_stat, p_value = stats.ttest_ind(dev_vals, reg_vals, equal_var=False)
                t_stat, p_value = stats.mannwhitneyu(dev_vals, reg_vals, alternative='two-sided')
                print(f"Time: {time}, p-value: {p_value}")

                # Add stars for significance level
                if p_value < 0.001:
                    star_label = '***'
                elif p_value < 0.01:
                    star_label = '**'
                elif p_value < 0.05:
                    star_label = '*'
                else:
                    star_label = ''

                if star_label:
                    p.text(x=[time], y=[max(np.max(dev_vals), np.max(reg_vals)) * 1.05], 
                           text=[star_label], text_font_size='16pt', text_color='black')

    if show_n:
        for time in all_times:
            n_dev = len(grouped_dev.get_group(time)[y_col]) if time in grouped_dev.groups else 0
            n_reg = len(grouped_reg.get_group(time)[y_col]) if time in grouped_reg.groups else 0
            y_pos=max([max(grouped_dev.get_group(time)[y_col]),max(grouped_reg.get_group(time)[y_col])])*1.1
            # Show number of samples for Development
            if n_dev > 0:
                p.text(x=[time - width / 2], y=y_pos,
                       text=[f"n={n_dev}"], text_font_size="10pt", text_color="blue")

            # Show number of samples for Regeneration
            if n_reg > 0:
                p.text(x=[time + width / 2], y=y_pos,
                       text=[f"n={n_reg}"], text_font_size="10pt", text_color="orange")


    # Box or violin plot for Development and Regeneration
    for condition, times, values, color in zip(['Development', 'Regeneration'],
                                               [times_dev, times_reg],
                                               [values_dev, values_reg],
                                               ['blue', 'orange']):
        if style == 'box':
            # Add box plots
            for i, (time, val) in enumerate(zip(times, values)):
                lower_bound, q1, q2, q3, upper_bound = np.percentile(val, [10, 25, 50, 75, 90])

                # Box for quartiles
                if condition=='Development':
                    time_shift=time-width/4 
                else:
                    time_shift=time+width/4 
                p.vbar(x=time_shift, width=width/2, bottom=q1, top=q3, color=color, alpha=0.4)
                  
        elif style == 'violin':
            # Add violin plots
            for i, (time, val) in enumerate(zip(times, values)):
                kde = gaussian_kde(val, bw_method=0.3)
                x = np.linspace(min(val)*0.8, max(val)*1.2, 100)
                kde_values = kde(x)
                kde_values = kde_values / kde_values.max() * width * 0.5
                
                if condition=='Development':
                    #time_shift=time-width/4 
                    p.patch(np.concatenate([time - kde_values, np.array([time,time])]),
                        np.concatenate([x, np.array([x[-1],x[0]])]),
                        alpha=0.3, color=color, line_color=color)
                else:
                    #time_shift=time+width/4 

                    p.patch(np.concatenate([np.array([time,time]), (time + kde_values)[::-1]]),
                            np.concatenate([np.array([x[-1],x[0]]), x[::-1]]),
                            alpha=0.3, color=color, line_color=color)

        # Add lines and whiskers (Development and Regeneration)
        if tight_style:
            for i, (time, val) in enumerate(zip(times, values)):
                lower_bound, q1, q2, q3, upper_bound = np.percentile(val, [10, 25, 50, 75, 90])
                print(lower_bound, q1, q2, q3, upper_bound)
                if condition=='Development':
                    # Whiskers
                    p.segment(x0=[time], y0=[lower_bound], x1=[time], y1=[upper_bound], color='black', line_width=2)
                    p.line(x=[time - width / 3, time ], y=[lower_bound, lower_bound], line_width=2, color='black')
                    p.line(x=[time - width / 3, time ], y=[upper_bound, upper_bound], line_width=2, color='black')

                    # Median
                    p.line(x=[time - width / 2, time], y=[q2, q2], line_width=3, color='black')
                    p.scatter(x=[time], y=[q2], size=10, color='black', marker='x')
                else:
                    # Whiskers
                    p.segment(x0=[time], y0=[lower_bound], x1=[time], y1=[upper_bound], color='black', line_width=2)
                    p.line(x=[time , time + width / 3], y=[lower_bound, lower_bound], line_width=2, color='black')
                    p.line(x=[time , time + width / 3], y=[upper_bound, upper_bound], line_width=2, color='black')

                    # Median
                    p.line(x=[time , time + width / 2], y=[q2, q2], line_width=3, color='black')
                    p.scatter(x=[time], y=[q2], size=10, color='black', marker='x')
        else:
            for i, (time, val) in enumerate(zip(times, values)):
                lower_bound, q1, q2, q3, upper_bound = np.percentile(val, [10, 25, 50, 75, 90])
                print(lower_bound, q1, q2, q3, upper_bound)
                if condition=='Development':
                    time_shift=time-width/4 
                else:
                    time_shift=time+width/4 

                # Whiskers
                p.segment(x0=[time_shift], y0=[lower_bound], x1=[time_shift], y1=[upper_bound], color='black', line_width=2)
                p.line(x=[time_shift - width / 6, time_shift + width / 6], y=[lower_bound, lower_bound], line_width=2, color='black')
                p.line(x=[time_shift - width / 6, time_shift + width / 6], y=[upper_bound, upper_bound], line_width=2, color='black')

                # Median
                p.line(x=[time_shift - width / 4, time_shift + width / 4], y=[q2, q2], line_width=3, color='black')
                p.scatter(x=[time_shift], y=[q2], size=10, color='black', marker='x')



    # Configure the legend
    p.legend.location = "top_left"

    # Add hover tool for scatter plot
    hover = HoverTool(renderers=[scatter_dev,scatter_reg])
    hover.tooltips = [
        ("Time (hpf)", "@{time in hpf}"),
        (y_name, f"@{{{y_col}}}"),
        ("Condition", "@{condition}"),
        ("Mask Folder", "@{Mask Folder}")
    ]
    p.add_tools(hover)

    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.legend.label_text_font_size = "12pt"
    p.legend.title_text_font_size = "14pt"

    # Show the plot
    show(p)

def plot_explanation(lower_bound, q1, q2, q3, upper_bound, plot_type='whiskers', title="Explanation Plot"):
    """
    Create a simple explanation plot to visualize whiskers, box, or violin plots with labeled percentiles.
    
    Parameters:
    -----------
    lower_bound : float
        The lower bound (10th percentile).
    q1 : float
        The first quartile (25th percentile).
    q2 : float
        The median (50th percentile).
    q3 : float
        The third quartile (75th percentile).
    upper_bound : float
        The upper bound (90th percentile).
    plot_type : str, optional (default='whiskers')
        The plot type: 'whiskers', 'box', or 'violin'.
    title : str, optional (default='Explanation Plot')
        Title of the plot.

    Returns:
    --------
    None
    Displays an interactive explanation plot in the browser.
    """

    # Create figure without axes, title, and grid lines
    p = figure(title=title, width=200, height=250, x_range=(-0.3, 0.5))
    p.axis.visible = False
    p.grid.visible = False
    
    # Define the time_shift as a constant, since there's only one plot element
    time_shift = 0
    # Whiskers
    p.segment(x0=[time_shift], y0=[lower_bound], x1=[time_shift], y1=[upper_bound], color='black', line_width=2)
    p.line(x=[time_shift - 0.1, time_shift + 0.1], y=[lower_bound, lower_bound], line_width=2, color='black')
    p.line(x=[time_shift - 0.1, time_shift + 0.1], y=[upper_bound, upper_bound], line_width=2, color='black')
    # Median
    p.line(x=[time_shift - 0.2, time_shift + 0.2], y=[q2, q2], line_width=3, color='black')
    p.scatter(x=[time_shift], y=[q2], size=10, color='black', marker='x')
    if plot_type == 'box':
        # Box plot (quartiles)
        if plot_type == 'box':
            p.vbar(x=time_shift, width=0.2, bottom=q1, top=q3, color="gray", alpha=0.4)
        
    elif plot_type == 'violin':
        # Simulate some sample data based on percentiles
        data = np.random.normal(loc=q2, scale=(upper_bound - lower_bound) / 4, size=10)
        kde = gaussian_kde(data, bw_method=0.3)
        y_vals = np.linspace(lower_bound * 0.7, upper_bound * 1.3, 100)
        kde_values = kde(y_vals)

        # Normalize the kde_values to fit the desired width in screen units
        kde_values = kde_values / kde_values.max() * 0.2

        # Draw a violin plot (mirrored KDE)
        p.patch(np.concatenate([time_shift - kde_values, (time_shift + kde_values)[::-1]]),
                np.concatenate([y_vals, y_vals[::-1]]),
                alpha=0.3, color='gray', line_color='gray')

    # Add labels for percentiles
    p.text(x=[time_shift + 0.2, time_shift + 0.2, time_shift + 0.2],
           y=[lower_bound, q2, upper_bound],
           text=['10th %',  '50th %',  '90th %'], text_align="left", text_baseline="middle")
    
    if plot_type == 'box':
        p.text(x=[time_shift + 0.2, time_shift + 0.2],
           y=[q1, q3],
           text=['25th %', '75th %'], text_align="left", text_baseline="middle")

    # Show the plot
    show(p)

def plot_histograms_bokeh(times, categories, meshes, quantities_to_plot=None, combinations=None, category_filter=None, time_filter=None):
    """
    Plot concatenated histograms of specified quantities from meshes, with options to combine times and categories,
    and with the ability to filter by category and/or time.

    Parameters:
    -----------
    times : list
        A list of time points corresponding to each mesh.
    categories : list
        A list of categories ('Development', 'Regeneration') corresponding to each mesh.
    meshes : list
        A list of pyvista PolyData meshes containing point data.
    quantities_to_plot : list or None
        List of quantities to be plotted. If None, all available quantities will be plotted.
    combinations : dict or None
        A dictionary specifying how to combine data:
        - 'combine_times_per_category': If True, combine all times for each category.
        - 'combine_categories_per_time': If True, combine both categories for each time.
        - If None, do not combine and plot each time-category pair separately.
    category_filter : list or None
        A list of categories to filter (e.g., ['Development', 'Regeneration']). If None, no filtering is applied.
    time_filter : tuple or None
        A tuple specifying the time range to filter (min_time, max_time). If None, no filtering is applied.

    Returns:
    --------
    None
    Displays an interactive hexbin density plot.
    """

    # Define the available quantities to be plotted
    available_quantities = ['mean_curvature_avg', 'gauss_curvature_avg', 'thickness_avg', 
                            'mean_curvature_std', 'gauss_curvature_std', 'thickness_std']
    
    # Use only specified quantities or default to all if none are provided
    if quantities_to_plot is None:
        quantities_to_plot = available_quantities
    else:
        # Filter only the valid quantities from the provided list
        quantities_to_plot = [q for q in quantities_to_plot if q in available_quantities]
        if not quantities_to_plot:
            raise ValueError(f"No valid quantities selected. Available quantities: {available_quantities}")
    
    # Normalize time values for color coding
    unique_times = sorted(set(times))
    time_colors = {time: Viridis256[int(255 * i / (len(unique_times) - 1))] for i, time in enumerate(unique_times)}
    
    # Define line styles for different conditions
    line_styles = {
        'Development': 'solid',
        'Regeneration': 'dashed'
    }
    
    # Apply filters if provided
    filtered_indices = list(range(len(meshes)))  # Default: no filter applied, include all

    if category_filter:
        filtered_indices = [i for i in filtered_indices if categories[i] in category_filter]

    if time_filter:
        min_time, max_time = time_filter
        filtered_indices = [i for i in filtered_indices if min_time <= times[i] <= max_time]
    
    # Prepare data based on combination settings
    combined_data = []
    
    if combinations is not None and combinations.get('combine_times_per_category', False):
        # Combine all times for each category
        for category in ['Development', 'Regeneration']:
            combined_data.append({
                'time': 'All Times',
                'category': category,
                'meshes': [meshes[i].point_data for i in filtered_indices if categories[i] == category]
            })
    elif combinations is not None and combinations.get('combine_categories_per_time', False):
        # Combine both categories for each time
        for time in unique_times:
            combined_data.append({
                'time': time,
                'category': 'Both',
                'meshes': [meshes[i].point_data for i in filtered_indices if times[i] == time]
            })
    elif combinations is not None and combinations.get('combine_times_per_category', False) and combinations.get('combine_categories_per_time', False):
        # Combine all times and both categories into one dataset
        combined_data.append({
            'time': 'All Times',
            'category': 'Both',
            'meshes': [meshes[i].point_data for i in filtered_indices]
        })
    else:
        # No combination: use individual time-category pairs
        for i in filtered_indices:
            combined_data.append({
                'time': times[i],
                'category': categories[i],
                'meshes': [meshes[i].point_data]
            })
    # Create a list to hold figures for gridplot
    figures = []
    
    # Create histograms for each selected quantity
    for quantity in quantities_to_plot:
        p = figure(title=f"Histogram of {quantity}",
                   x_axis_label=quantity, y_axis_label='Density', width=1000, height=600)
        
        # Enable hover tool
        hover = HoverTool(tooltips=[
            ("Time", "@time"),
            ("Condition", "@condition")
        ])
        p.add_tools(hover)
        
        # Plot histograms based on the prepared data
        for data in combined_data:
            concatenated_data = []
            
            # Concatenate data for the current quantity across all meshes in the group
            for mesh_point_data in data['meshes']:
                if quantity in mesh_point_data:
                    concatenated_data.extend(mesh_point_data[quantity])
            
            if concatenated_data:
                # Create histogram data
                hist, edges = np.histogram(concatenated_data, bins=30, density=True)
                
                # Create a ColumnDataSource for the Bokeh plot
                source = ColumnDataSource(data={
                    'top': hist,
                    'left': edges[:-1],
                    'right': edges[1:],
                    'time': [data['time']] * len(hist),
                    'condition': [data['category']] * len(hist)
                })
                
                # Use color for time and line style for category
                color = time_colors[data['time']] if data['time'] != 'All Times' else 'black'
                line_dash = line_styles[data['category']] if data['category'] != 'Both' else 'solid'
                
                # Add the histogram to the plot
                p.quad(top='top', bottom=0, left='left', right='right',
                       line_color=color, fill_color=None,
                       line_dash=line_dash, line_width=2, source=source,
                       legend_label=f'{data["category"]} (Time: {data["time"]})')
        
        # Add to figures list for gridplot
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        figures.append(p)
    
    # Arrange plots in a grid (pass figures directly, no nested list)
    grid = gridplot(figures, ncols=1)
    
    # Show the grid of histograms
    show(grid)

def plot_scatter_quantities(times, categories, meshes, x_quantity, y_quantity, mode='category'):
    """
    Create a scatter plot of two specified quantities using Bokeh, with color-coding and hover tools.

    Parameters:
    -----------
    times : list
        A list of time points corresponding to each mesh.
    categories : list
        A list of categories ('Development', 'Regeneration') corresponding to each mesh.
    meshes : list
        A list of pyvista PolyData meshes containing point data.
    x_quantity : str
        The quantity to plot on the x-axis (e.g., 'gauss_curvature_avg').
    y_quantity : str
        The quantity to plot on the y-axis (e.g., 'mean_curvature_avg').
    mode : str, optional (default='category')
        Determines how the plot is color-coded and shaped.
        'category' : Colors the plot based on the 'condition' (Development or Regeneration).
        'time'     : Colors the plot based on 'time' and shapes by the 'condition'.

    Returns:
    --------
    None
    Displays an interactive scatter plot.
    """
    
    # Extract the relevant data
    data = {
        'x': [],
        'y': [],
        'time': [],
        'condition': []
    }

    for i, mesh in enumerate(meshes):
        if x_quantity in mesh.point_data and y_quantity in mesh.point_data:
            x_data = mesh.point_data[x_quantity]
            y_data = mesh.point_data[y_quantity]
            n_points = len(x_data)
            
            # Append the data
            data['x'].extend(x_data)
            data['y'].extend(y_data)
            data['time'].extend([times[i]] * n_points)
            data['condition'].extend([categories[i]] * n_points)

    # Create a ColumnDataSource for Bokeh
    source = ColumnDataSource(data=data)

    # Define color palettes and marker shapes
    colors = {'Development': 'blue', 'Regeneration': 'orange'}
    shapes = {'Development': 'circle', 'Regeneration': 'triangle'}

    # Create the figure
    p = figure(title=f"{x_quantity} vs {y_quantity}",
               x_axis_label=x_quantity,
               y_axis_label=y_quantity,
               width=1000, height=800,
               tools="pan,wheel_zoom,box_zoom,reset,save")

    # Plot based on mode
    if mode == 'category':  # Color by category (Development/Regeneration)
        for condition, color in colors.items():
            shape = shapes[condition]
            subset = source.to_df()[source.to_df()['condition'] == condition]
            source_subset = ColumnDataSource(subset)
            
            # Scatter plot for each condition with specific color and shape
            p.scatter(x='x', y='y', source=source_subset,
                      size=10, color=color, marker=shape,
                      legend_label=condition, alpha=0.6)

    elif mode == 'time':  # Color by time, shape by condition
        time_mapper = linear_cmap(field_name='time', palette=Viridis256,
                                  low=min(data['time']), high=max(data['time']))
        
        for condition, shape in shapes.items():
            subset = source.to_df()[source.to_df()['condition'] == condition]
            source_subset = ColumnDataSource(subset)
            
            # Scatter plot with color mapped by time and shaped by condition
            p.scatter(x='x', y='y', source=source_subset,
                      size=10, color=time_mapper, marker=shape,
                      legend_label=condition, alpha=0.6)

    # Add hover tool
    hover = HoverTool()
    hover.tooltips = [
        (x_quantity, "@x"),
        (y_quantity, "@y"),
        ("Time", "@time"),
        ("Condition", "@condition")
    ]
    p.add_tools(hover)

    # Configure legend
    p.legend.title = 'Condition'
    p.legend.location = "top_left"

    # Show the plot
    show(p)

def plot_density_hexbin(times, categories, meshes, x_quantity, y_quantity, rel_size=0.001, rel_aspect_scale=1.0, 
                        category_filter=None, time_filter=None, color_by_category=True,show_hist=True,plot_k_kurves=False):
    """
    Create a hexbin density plot of two specified quantities using Bokeh, with color-coding and hover tools.
    Optionally, filter data by category ('Development', 'Regeneration') and/or by time (given by an interval),
    and color-code scatter points by category.
    
    Parameters:
    -----------
    times : list
        A list of time points corresponding to each mesh.
    categories : list
        A list of categories ('Development', 'Regeneration') corresponding to each mesh.
    meshes : list
        A list of pyvista PolyData meshes containing point data.
    x_quantity : str
        The quantity to plot on the x-axis (e.g., 'gauss_curvature_avg').
    y_quantity : str
        The quantity to plot on the y-axis (e.g., 'mean_curvature_avg').
    rel_size : float, optional (default=0.001)
        The relative size of the hexagons in the hexbin plot.
    rel_aspect_scale : float, optional (default=1.0)
        The aspect scale to adjust the hexbin aspect ratio.
    category_filter : list, optional (default=None)
        A list of categories to filter by (e.g., ['Development', 'Regeneration']). If None, no filtering is applied.
    time_filter : tuple, optional (default=None)
        A tuple specifying the time interval to filter by (e.g., (48, 144)). If None, no filtering is applied.
    color_by_category : bool, optional (default=True)
        Whether to color-code scatter points by category ('Development' in blue, 'Regeneration' in orange).

    Returns:
    --------
    None
    Displays an interactive hexbin density plot.
    """
    
    # Extract the relevant data
    x_data = []
    y_data = []
    colors = []

    for i, mesh in enumerate(meshes):
        # Apply category filter if provided
        if category_filter is not None and categories[i] not in category_filter:
            continue
        
        # Apply time filter if provided
        if time_filter is not None and not (time_filter[0] <= times[i] <= time_filter[1]):
            continue
        
        if x_quantity in mesh.point_data and y_quantity in mesh.point_data:
            x_data.extend(mesh.point_data[x_quantity])
            y_data.extend(mesh.point_data[y_quantity])
            
            # Assign colors based on category
            if color_by_category:
                color = 'blue' if categories[i] == 'Development' else 'orange'
                colors.extend([color] * len(mesh.point_data[x_quantity]))

    # Convert to NumPy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Trim data to 1st and 99th percentiles for both x and y axes
    x_min, x_max = np.percentile(x_data, [5, 95])
    y_min, y_max = np.percentile(y_data, [5, 95])

    # Calculate size of hexagons relative to the data range
    size = rel_size * np.sqrt((x_max - x_min) * (y_max - y_min))

    # Calculate aspect scale if not provided
    aspect_scale = rel_aspect_scale * (y_max - y_min) / (x_max - x_min)

    # Calculate plot dimensions
    plot_width = 700
    plot_height = 700

    # Create the figure with trimmed ranges
    p = figure(title=f"{x_quantity} vs {y_quantity} (Hexbin Density)",
               x_axis_label=x_quantity, y_axis_label=y_quantity,
               width=plot_width, height=plot_height,
               x_range=(x_min, x_max), y_range=(y_min, y_max),
               tools="pan,wheel_zoom,box_zoom,reset,save")

    if show_hist:
        bins = p.hexbin(x_data, y_data, size=size, hover_color="pink", hover_alpha=0.8, aspect_scale=aspect_scale)
        # Add hover tool for hexbin counts
        hover = HoverTool(tooltips=[("Count", "@c")], mode='mouse', point_policy='follow_mouse')
        p.add_tools(hover)

    # Add scatter points, color-coded by category if selected
    if color_by_category:
        p.circle(x_data, y_data, color=colors, size=3)
    else:
        p.circle(x_data, y_data, color="black", size=1)

    if plot_k_kurves:
        # Plot the curve y = x^2 from x_min to x_max
        x_vals = np.linspace(x_min*2, x_max*2, 100)
        y_vals = x_vals**2
        p.line(x_vals, y_vals, line_width=2, color="black")

        # Plot the vertical line x = 0 from y_min to 0
        p.line([0, 0], [np.min(y_data), 0], line_width=2, color="black")

    # Show the plot
    show(p)


def plot_compare_timeseries(df1, df2, category='Development', y_col=None, style='box', y_scaling=1.0, y_name=None,
                            test_significance=True, show_n=True, y0=None, tight_style=True, 
                            label_df1="DataFrame 1", label_df2="DataFrame 2"):
    """
    Compare two time-series dataframes for a specific category using box plots or violin plots.
    
    Parameters:
    -----------
    df1, df2 : pd.DataFrame
        The input DataFrames containing the data to plot.
    category : str
        The category to compare (e.g., 'Development', 'Regeneration').
    y_col : str
        The column name from `df1` and `df2` to use for the y-axis (quantity of interest).
    style : str, optional (default='box')
        The plot style: 'box' for box plot or 'violin' for violin plot.
    label_df1, label_df2 : str, optional (default='DataFrame 1', 'DataFrame 2')
        Labels to display in the plot legend for the two dataframes.

    Returns:
    --------
    None
    Displays an interactive time-series plot with scatter points and optional fit results.
    """
    
    if y_col is None:
        raise ValueError("You must specify the column for the y-axis (y_col).")
    
    if y_name is None:
        y_name = y_col
    
    # Scale the data
    df1 = df1.copy()
    df2 = df2.copy()
    df1[y_col] = df1[y_col] * y_scaling
    df2[y_col] = df2[y_col] * y_scaling

    # Filter data for the specific category
    df1_filtered = df1[df1['condition'] == category].copy()
    df2_filtered = df2[df2['condition'] == category].copy()

    # Ensure 'time in hpf' exists and is integer
    df1_filtered['time in hpf'] = df1_filtered['time in hpf'].astype(int)
    df2_filtered['time in hpf'] = df2_filtered['time in hpf'].astype(int)

    # Group data by 'time in hpf' for both dataframes
    grouped_df1 = df1_filtered.groupby('time in hpf')
    grouped_df2 = df2_filtered.groupby('time in hpf')

    # Extract unique times and prepare data
    times_df1 = sorted(list(grouped_df1.groups.keys()))
    values_df1 = [grouped_df1.get_group(time)[y_col].values for time in times_df1]
    times_df2 = sorted(list(grouped_df2.groups.keys()))
    values_df2 = [grouped_df2.get_group(time)[y_col].values for time in times_df2]

    # Define plot width
    all_times = sorted(set(times_df1).union(set(times_df2)))
    width = 0.8 * min(np.diff(all_times)) if len(all_times) > 1 else 1
    
    # Apply the shift for scatter points
    df1_filtered['shifted_time'] = df1_filtered['time in hpf'] - width * 0.0
    df2_filtered['shifted_time'] = df2_filtered['time in hpf'] + width * 0.0

    # Create a ColumnDataSource for scatter plot
    scatter_source_df1 = ColumnDataSource(df1_filtered)
    scatter_source_df2 = ColumnDataSource(df2_filtered)

    max_y_value = max(df1_filtered[y_col].max(), df2_filtered[y_col].max()) * 1.2

    # Create figure with a shared numerical x-axis
    if y0 is None:
        # Let Bokeh automatically decide the y-range
        p = figure(title=f"Time-series Comparison ({style.capitalize()} Plot) of {y_name} by Time",
                   x_axis_label="Time (hpf)",
                   y_axis_label=y_name,
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=1000, height=600)
    else:
        # Set the y-range manually if y0 is specified
        p = figure(title=f"Time-series Comparison ({style.capitalize()} Plot) of {y_name} by Time",
                   x_axis_label="Time (hpf)",
                   y_axis_label=y_name,
                   y_range=(y0, max_y_value),  # Set both the lower and upper bounds
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=1000, height=600)

    # Scatter plot for individual points (df1 and df2)
    scatter_df1 = p.scatter(x='shifted_time', y=y_col, source=scatter_source_df1,
                            size=8, color='blue', alpha=0.6, legend_label=label_df1)
    scatter_df2 = p.scatter(x='shifted_time', y=y_col, source=scatter_source_df2,
                            size=8, color='green', alpha=0.6, legend_label=label_df2)

    
    # Test significance
    if test_significance:
        for time in all_times:
            if time in grouped_df1.groups.keys() and time in grouped_df2.groups.keys():
                vals_df1 = grouped_df1.get_group(time)[y_col].values
                vals_df2 = grouped_df2.get_group(time)[y_col].values

                # Perform Mann-Whitney U test
                t_stat, p_value = mannwhitneyu(vals_df1, vals_df2, alternative='two-sided')
                print(f"Time: {time}, p-value: {p_value}")

                # Add stars for significance level
                if p_value < 0.001:
                    star_label = '***'
                elif p_value < 0.01:
                    star_label = '**'
                elif p_value < 0.05:
                    star_label = '*'
                else:
                    star_label = ''

                if star_label:
                    p.text(x=[time], y=[max(np.max(vals_df1), np.max(vals_df2)) * 1.05], 
                           text=[star_label], text_font_size='16pt', text_color='black')

    # Box or violin plot for DataFrame 1 and DataFrame 2
    for df, times, values, color, label in zip([df1_filtered, df2_filtered],
                                               [times_df1, times_df2],
                                               [values_df1, values_df2],
                                               ['blue', 'green'],
                                               [label_df1, label_df2]):
        if style == 'box':
            # Add box plots
            for i, (time, val) in enumerate(zip(times, values)):
                lower_bound, q1, q2, q3, upper_bound = np.percentile(val, [10, 25, 50, 75, 90])

                # Box for quartiles
                time_shift = time - width / 4 if color == 'blue' else time + width / 4
                p.vbar(x=time_shift, width=width / 2, bottom=q1, top=q3, color=color, alpha=0.4)

        elif style == 'violin':
            # Add violin plots
            for i, (time, val) in enumerate(zip(times, values)):
                kde = gaussian_kde(val, bw_method=0.3)
                x = np.linspace(min(val)*0.8, max(val)*1.2, 100)
                kde_values = kde(x)
                kde_values = kde_values / kde_values.max() * width * 0.5
                
                if color == 'blue':
                    p.patch(np.concatenate([time - kde_values, np.array([time, time])]),
                            np.concatenate([x, np.array([x[-1], x[0]])]),
                            alpha=0.3, color=color, line_color=color)
                else:
                    p.patch(np.concatenate([np.array([time, time]), (time + kde_values)[::-1]]),
                            np.concatenate([np.array([x[-1], x[0]]), x[::-1]]),
                            alpha=0.3, color=color, line_color=color)

        # Add lines and whiskers (Development and Regeneration)
        if tight_style:
            for i, (time, val) in enumerate(zip(times, values)):
                lower_bound, q1, q2, q3, upper_bound = np.percentile(val, [10, 25, 50, 75, 90])
                print(lower_bound, q1, q2, q3, upper_bound)
                if color=='blue':
                    # Whiskers
                    p.segment(x0=[time], y0=[lower_bound], x1=[time], y1=[upper_bound], color='black', line_width=2)
                    p.line(x=[time - width / 3, time ], y=[lower_bound, lower_bound], line_width=2, color='black')
                    p.line(x=[time - width / 3, time ], y=[upper_bound, upper_bound], line_width=2, color='black')

                    # Median
                    p.line(x=[time - width / 2, time], y=[q2, q2], line_width=3, color='black')
                    p.scatter(x=[time], y=[q2], size=10, color='black', marker='x')
                else:
                    # Whiskers
                    p.segment(x0=[time], y0=[lower_bound], x1=[time], y1=[upper_bound], color='black', line_width=2)
                    p.line(x=[time , time + width / 3], y=[lower_bound, lower_bound], line_width=2, color='black')
                    p.line(x=[time , time + width / 3], y=[upper_bound, upper_bound], line_width=2, color='black')

                    # Median
                    p.line(x=[time , time + width / 2], y=[q2, q2], line_width=3, color='black')
                    p.scatter(x=[time], y=[q2], size=10, color='black', marker='x')
        else:
            for i, (time, val) in enumerate(zip(times, values)):
                lower_bound, q1, q2, q3, upper_bound = np.percentile(val, [10, 25, 50, 75, 90])
                print(lower_bound, q1, q2, q3, upper_bound)
                if color=='blue':
                    time_shift=time-width/4 
                else:
                    time_shift=time+width/4 

                # Whiskers
                p.segment(x0=[time_shift], y0=[lower_bound], x1=[time_shift], y1=[upper_bound], color='black', line_width=2)
                p.line(x=[time_shift - width / 6, time_shift + width / 6], y=[lower_bound, lower_bound], line_width=2, color='black')
                p.line(x=[time_shift - width / 6, time_shift + width / 6], y=[upper_bound, upper_bound], line_width=2, color='black')

                # Median
                p.line(x=[time_shift - width / 4, time_shift + width / 4], y=[q2, q2], line_width=3, color='black')
                p.scatter(x=[time_shift], y=[q2], size=10, color='black', marker='x')

    # Configure the legend
    p.legend.location = "top_left"

    # Add hover tool for scatter plot
    hover = HoverTool(renderers=[scatter_df1, scatter_df2])
    hover.tooltips = [
        ("Time (hpf)", "@{time in hpf}"),
        (y_name, f"@{{{y_col}}}"),
        ("Condition", "@{condition}"),
        ("Mask Folder", "@{Mask Folder}")
    ]
    p.add_tools(hover)

    # Show the plot
    show(p)

def plot_side_by_side_timeseries(df1, df2, y_col=None, style='box', y_scaling=1.0, y_name=None,
                                 test_significance=True, show_n=True, tight_style=True,
                                 label_df1="Experiment 1", label_df2="Experiment 2"):
    """
    Compare 'Development' and 'Regeneration' side by side for two time-series dataframes.
    
    Parameters:
    -----------
    df1, df2 : pd.DataFrame
        The input DataFrames containing the data to plot.
    y_col : str
        The column name from `df1` and `df2` to use for the y-axis (quantity of interest).
    style : str, optional (default='box')
        The plot style: 'box' for box plot or 'violin' for violin plot.
    label_df1, label_df2 : str, optional (default='Experiment 1', 'Experiment 2')
        Labels to display in the plot legend for the two experiments.

    Returns:
    --------
    None
    Displays an interactive time-series plot with side-by-side comparisons for 'Development' and 'Regeneration'.
    """
    
    if y_col is None:
        raise ValueError("You must specify the column for the y-axis (y_col).")
    
    if y_name is None:
        y_name = y_col
    
    # Scale the data
    df1 = df1.copy()
    df2 = df2.copy()
    df1[y_col] = df1[y_col] * y_scaling
    df2[y_col] = df2[y_col] * y_scaling

    # Find maximum y-value for consistent y-axis scaling
    max_y_value = max(df1[y_col].max(), df2[y_col].max()) * 1.2
    y_range = (0, max_y_value)

    categories = ['Development', 'Regeneration']
    
    plots = []
    for category in categories:
        # Filter data for the specific category
        df1_filtered = df1[df1['condition'] == category].copy()
        df2_filtered = df2[df2['condition'] == category].copy()

        # Ensure 'time in hpf' exists and is integer
        df1_filtered['time in hpf'] = df1_filtered['time in hpf'].astype(int)
        df2_filtered['time in hpf'] = df2_filtered['time in hpf'].astype(int)

        # Group data by 'time in hpf' for both dataframes
        grouped_df1 = df1_filtered.groupby('time in hpf')
        grouped_df2 = df2_filtered.groupby('time in hpf')

        # Extract unique times and prepare data
        times_df1 = sorted(list(grouped_df1.groups.keys()))
        values_df1 = [grouped_df1.get_group(time)[y_col].values for time in times_df1]
        times_df2 = sorted(list(grouped_df2.groups.keys()))
        values_df2 = [grouped_df2.get_group(time)[y_col].values for time in times_df2]

        # Define plot width
        all_times = sorted(set(times_df1).union(set(times_df2)))
        width = 0.8 * min(np.diff(all_times)) if len(all_times) > 1 else 1

        # Apply the shift for scatter points
        df1_filtered['shifted_time'] = df1_filtered['time in hpf'] - width * 0.0
        df2_filtered['shifted_time'] = df2_filtered['time in hpf'] + width * 0.0

        # Create a ColumnDataSource for scatter plot
        scatter_source_df1 = ColumnDataSource(df1_filtered)
        scatter_source_df2 = ColumnDataSource(df2_filtered)

        
 

        # Create figure with a shared numerical x-axis
        p = figure(title=f"{category} Comparison ({style.capitalize()} Plot) of {y_name} by Time",
                   x_axis_label="Time (hpf)",
                   y_axis_label=y_name,
                   y_range=y_range,
                   tools="pan,wheel_zoom,box_zoom,reset,save",
                   width=1000, height=600)

        # Scatter plot for individual points (df1 and df2)
        scatter_df1 = p.scatter(x='shifted_time', y=y_col, source=scatter_source_df1,
                                size=8, color='blue', alpha=0.6, legend_label=label_df1)
        scatter_df2 = p.scatter(x='shifted_time', y=y_col, source=scatter_source_df2,
                                size=8, color='green', alpha=0.6, legend_label=label_df2)

        # Test significance
        if test_significance:
            for time in all_times:
                if time in grouped_df1.groups.keys() and time in grouped_df2.groups.keys():
                    vals_df1 = grouped_df1.get_group(time)[y_col].values
                    vals_df2 = grouped_df2.get_group(time)[y_col].values

                    # Perform Mann-Whitney U test
                    t_stat, p_value = mannwhitneyu(vals_df1, vals_df2, alternative='two-sided')
                    print(f"Time: {time}, p-value: {p_value}")

                    # Add stars for significance level
                    if p_value < 0.001:
                        star_label = '***'
                    elif p_value < 0.01:
                        star_label = '**'
                    elif p_value < 0.05:
                        star_label = '*'
                    else:
                        star_label = ''

                    if star_label:
                        p.text(x=[time], y=[max(np.max(vals_df1), np.max(vals_df2)) * 1.05], 
                               text=[star_label], text_font_size='16pt', text_color='black')

        # Box or violin plot for DataFrame 1 and DataFrame 2
        for df, times, values, color, label in zip([df1_filtered, df2_filtered],
                                                   [times_df1, times_df2],
                                                   [values_df1, values_df2],
                                                   ['blue', 'green'],
                                                   [label_df1, label_df2]):
            if style == 'box':
                # Add box plots
                for i, (time, val) in enumerate(zip(times, values)):
                    lower_bound, q1, q2, q3, upper_bound = np.percentile(val, [10, 25, 50, 75, 90])

                    # Box for quartiles
                    time_shift = time - width / 4 if color == 'blue' else time + width / 4
                    p.vbar(x=time_shift, width=width / 2, bottom=q1, top=q3, color=color, alpha=0.4)

            elif style == 'violin':
                # Add violin plots
                for i, (time, val) in enumerate(zip(times, values)):
                    kde = gaussian_kde(val, bw_method=0.3)
                    x = np.linspace(min(val) * 0.8, max(val) * 1.2, 100)
                    kde_values = kde(x)
                    kde_values = kde_values / kde_values.max() * width * 0.5

                    if color == 'blue':
                        p.patch(np.concatenate([time - kde_values, np.array([time, time])]),
                                np.concatenate([x, np.array([x[-1], x[0]])]),
                                alpha=0.3, color=color, line_color=color)
                    else:
                        p.patch(np.concatenate([np.array([time, time]), (time + kde_values)[::-1]]),
                                np.concatenate([np.array([x[-1], x[0]]), x[::-1]]),
                                alpha=0.3, color=color, line_color=color)


                    # Add lines and whiskers (Development and Regeneration)
            if tight_style:
                for i, (time, val) in enumerate(zip(times, values)):
                    lower_bound, q1, q2, q3, upper_bound = np.percentile(val, [10, 25, 50, 75, 90])
                    print(lower_bound, q1, q2, q3, upper_bound)
                    if color=='blue':
                        # Whiskers
                        p.segment(x0=[time], y0=[lower_bound], x1=[time], y1=[upper_bound], color='black', line_width=2)
                        p.line(x=[time - width / 3, time ], y=[lower_bound, lower_bound], line_width=2, color='black')
                        p.line(x=[time - width / 3, time ], y=[upper_bound, upper_bound], line_width=2, color='black')

                        # Median
                        p.line(x=[time - width / 2, time], y=[q2, q2], line_width=3, color='black')
                        p.scatter(x=[time], y=[q2], size=10, color='black', marker='x')
                    else:
                        # Whiskers
                        p.segment(x0=[time], y0=[lower_bound], x1=[time], y1=[upper_bound], color='black', line_width=2)
                        p.line(x=[time , time + width / 3], y=[lower_bound, lower_bound], line_width=2, color='black')
                        p.line(x=[time , time + width / 3], y=[upper_bound, upper_bound], line_width=2, color='black')

                        # Median
                        p.line(x=[time , time + width / 2], y=[q2, q2], line_width=3, color='black')
                        p.scatter(x=[time], y=[q2], size=10, color='black', marker='x')
            else:
                for i, (time, val) in enumerate(zip(times, values)):
                    lower_bound, q1, q2, q3, upper_bound = np.percentile(val, [10, 25, 50, 75, 90])
                    print(lower_bound, q1, q2, q3, upper_bound)
                    if color=='blue':
                        time_shift=time-width/4 
                    else:
                        time_shift=time+width/4 

                    # Whiskers
                    p.segment(x0=[time_shift], y0=[lower_bound], x1=[time_shift], y1=[upper_bound], color='black', line_width=2)
                    p.line(x=[time_shift - width / 6, time_shift + width / 6], y=[lower_bound, lower_bound], line_width=2, color='black')
                    p.line(x=[time_shift - width / 6, time_shift + width / 6], y=[upper_bound, upper_bound], line_width=2, color='black')

                    # Median
                    p.line(x=[time_shift - width / 4, time_shift + width / 4], y=[q2, q2], line_width=3, color='black')
                    p.scatter(x=[time_shift], y=[q2], size=10, color='black', marker='x')



        plots.append(p)

    # Arrange the two plots side by side
    show(row(*plots))

