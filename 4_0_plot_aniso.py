import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression

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

    print(cleaned_df.columns)


    # Define the color and marker dictionaries
    color_dict = {'Regeneration': 'orange',
                 'Development': 'blue', 
                 }
    marker_dict = {'Development': 'circle', 'Regeneration': 'triangle'}

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



if __name__ == "__main__":
    plot()

