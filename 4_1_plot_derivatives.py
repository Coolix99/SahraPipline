import pandas as pd
import os
import numpy as np

from plotHelper import plot_scatter_corner,plot_double_timeseries_II,add_data_to_plot_II,add_fit_to_plot_II
from bokeh.io import show
from bokeh.plotting import figure

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


from scipy.interpolate import UnivariateSpline

def fit_and_sample_derivatives(df: pd.DataFrame, column: str, N: int = 200,s: float = 0):
    results = {}
    grouped = df.groupby('condition')

    for condition, group in grouped:
        if condition == '4850cut' or condition == '48FF_cut' or condition == '72FF_cut':
            continue
        
        # Calculate mean and std for each time
        stats = group.groupby('time in hpf')[column].agg(['mean', 'std']).reset_index()

        # Create a dense time grid for interpolation
        dense_t = np.linspace(stats['time in hpf'].min(), stats['time in hpf'].max(), 500)

        
        condition_fits = {
            'fitted_values': [],
            'derivative': [],
            'relative_derivative': [],
        }
        for _ in range(N):
            # Sample around mean with std
            sampled_values = np.random.normal(stats['mean'], stats['std']/2)
            # Fit a UnivariateSpline
            
            spline = UnivariateSpline(stats['time in hpf'], sampled_values,w=1/stats['std'], s=s,k=2)
            fitted_values = spline(dense_t)
            
            # Calculate derivatives
            derivative = spline.derivative()(dense_t)
            relative_derivative = derivative / fitted_values
            
            
            condition_fits['fitted_values'].append(fitted_values)
            condition_fits['derivative'].append(derivative)
            condition_fits['relative_derivative'].append(relative_derivative)
        condition_fits['time']=dense_t
        results[condition] = condition_fits
    
    return results


def main():
    df=getData()
    print(df.columns)

    #region Surface Area

    res=fit_and_sample_derivatives(df, 'Surface Area', N=500,s=0.5e1)
    fit_results_dev = {
        't_values': res['Development']['time'],
        'fits': res['Development']['fitted_values']
    }
    fit_results_reg = {
        't_values': res['Regeneration']['time'],
        'fits': res['Regeneration']['fitted_values']
    }

    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Surface Area', style='violin',y_scaling=1,y_name=r'Area $$(\mu m)^2$$',test_significance=True,y0=0)
    # #show(p)
    # p = add_fit_to_plot_II(p, fit_results_dev, color='blue',label='Development')  
    # p = add_fit_to_plot_II(p, fit_results_reg, color='orange',label='Regeneration')  
    # show(p)

    # p = figure(title=f"Surface Area growth rate",
    #            x_axis_label=r'time in hpf',
    #            y_axis_label=r'dArea/dt $$(\mu m)^2/h$$',
    #            width=700, height=400,
    #            tools="pan,wheel_zoom,box_zoom,reset,save")
    # fit_results_dev = {
    #     't_values': res['Development']['time'],
    #     'fits': res['Development']['derivative']
    # }
    # fit_results_reg = {
    #     't_values': res['Regeneration']['time'],
    #     'fits': res['Regeneration']['derivative']
    # }
    # p = add_fit_to_plot_II(p, fit_results_dev, color='blue',label='Development')  
    # p = add_fit_to_plot_II(p, fit_results_reg, color='orange',label='Regeneration')  
    # show(p)
    # import time
    # time.sleep(2.5)
    # p = figure(title=f"relative Surface Area growth rate",
    #            x_axis_label=r'time in hpf',
    #            y_axis_label=r'dArea/dt/Area $$1/h$$',
    #            width=700, height=400,
    #            tools="pan,wheel_zoom,box_zoom,reset,save")
    # fit_results_dev = {
    #     't_values': res['Development']['time'],
    #     'fits': res['Development']['relative_derivative']
    # }
    # fit_results_reg = {
    #     't_values': res['Regeneration']['time'],
    #     'fits': res['Regeneration']['relative_derivative']
    # }
    # p = add_fit_to_plot_II(p, fit_results_dev, color='blue',label='Development')  
    # p = add_fit_to_plot_II(p, fit_results_reg, color='orange',label='Regeneration')  
    # show(p)
    
    # #endregion

    # #region Volume
    # res=fit_and_sample_derivatives(df, 'Volume', N=500,s=0.5e1)
    
    # fit_results_dev = {
    #     't_values': res['Development']['time'],
    #     'fits': res['Development']['fitted_values']
    # }
    # fit_results_reg = {
    #     't_values': res['Regeneration']['time'],
    #     'fits': res['Regeneration']['fitted_values']
    # }

    # p,width=plot_double_timeseries_II(df,categories=('Development','Regeneration'), y_col='Volume', style='violin',y_scaling=1,y_name=r'Volume $$(\mu m)^3$$',test_significance=True,y0=0)
    # show(p)
    # p = add_fit_to_plot_II(p, fit_results_dev, color='blue',label='Development')  
    # p = add_fit_to_plot_II(p, fit_results_reg, color='orange',label='Regeneration')  
    # show(p)

    # p = figure(title=f"Volume growth rate",
    #            x_axis_label=r'time in hpf',
    #            y_axis_label=r'dV/dt $$(\mu m)^3/h$$',
    #            width=700, height=400,
    #            tools="pan,wheel_zoom,box_zoom,reset,save")
    # fit_results_dev = {
    #     't_values': res['Development']['time'],
    #     'fits': res['Development']['derivative']
    # }
    # fit_results_reg = {
    #     't_values': res['Regeneration']['time'],
    #     'fits': res['Regeneration']['derivative']
    # }
    # p = add_fit_to_plot_II(p, fit_results_dev, color='blue',label='Development')  
    # p = add_fit_to_plot_II(p, fit_results_reg, color='orange',label='Regeneration')  
    # show(p)
    # import time
    # time.sleep(2.5)
    p = figure(title=f"relative Volume growth rate",
               x_axis_label=r'hpf [t]',
               y_axis_label=r'$$\dot{V}/V \quad [1/h]$$',
               width=700, height=400,
               tools="pan,wheel_zoom,box_zoom,reset,save")
    fit_results_dev = {
        't_values': res['Development']['time'],
        'fits': res['Development']['relative_derivative']
    }
    fit_results_reg = {
        't_values': res['Regeneration']['time'],
        'fits': res['Regeneration']['relative_derivative']
    }
    p = add_fit_to_plot_II(p, fit_results_dev, color='blue',label='Development')  
    p = add_fit_to_plot_II(p, fit_results_reg, color='orange',label='Regeneration')  
    p.xaxis.axis_label_text_font_size = "14pt"
    p.yaxis.axis_label_text_font_size = "14pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    show(p)
    #endregion

    # plot_single_timeseries(df, filter_col='condition', filter_value='Regeneration', y_col='Volume', style='violin', color='orange',width=None)
    # plot_double_timeseries(df, y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Tissue Volume $$10^6 \mu m^3$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='L PD', style='violin',y_scaling=1,y_name=r'$$L_{DP} in \mu m$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='L AP', style='violin',y_scaling=1,y_name=r'$$L_{AP} in \mu m$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='L DV', style='violin',y_scaling=1,y_name=r'$$L_{DV} in \mu m$$',test_significance=True,y0=0)
    
    
if __name__ == "__main__":
    main()