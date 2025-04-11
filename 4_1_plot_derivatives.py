import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from plotHelper.plotBokehHelper_old import plot_scatter_corner,plot_double_timeseries_II,add_data_to_plot_II,add_fit_to_plot_II
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
        if  condition == '48FF_cut' or condition == '72FF_cut':
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

def plot_fit_with_uncertainty(fit_results_list, colors, labels, title, xlabel, ylabel, ymin=0,ymax=None,xmax=None,xmin=None,dy=None,dx=None):
    """
    Plot fit results with uncertainty bands using Matplotlib.

    Parameters:
    -----------
    fit_results_list : list of dict
        List of fit result dictionaries. Each should have 't_values' and 'fits'.
    colors : list of str
        List of colors corresponding to each category.
    labels : list of str
        List of labels for the legend.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    """
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig = plt.figure(figsize=(6, 4))

    for fit_results, color, label in zip(fit_results_list, colors, labels):
        t_values = fit_results['t_values']
        fits = fit_results['fits']

        fit_lower2sig, fit_lowersig, fit_mean, fit_uppersig, fit_upper2sig = np.percentile(
            fits, [2.3, 15.9, 50, 84.1, 97.7], axis=0
        )

        plt.plot(t_values, fit_mean, color=color, linewidth=2, label=label)
        plt.fill_between(t_values, fit_lowersig, fit_uppersig, color=color, alpha=0.2)
        #plt.fill_between(t_values, fit_lower2sig, fit_upper2sig, color=color, alpha=0.1)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=22)

    if ymin is None:
        ymin = plt.ylim()[0]
    if ymax is None:
        ymax = plt.ylim()[1]
    if xmin is None:
        xmin = plt.xlim()[0]
    if xmax is None:
        xmax = plt.xlim()[1]
    if dx is None:
        dx = (xmax - xmin) / 10
    if dy is None:
        dy = (ymax - ymin) / 10
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    plt.xticks(np.arange(xmin, xmax, dx), fontsize=16)
    plt.yticks(np.arange(ymin, ymax, dy), fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    df=getData()
    print(df.columns)


    # res=fit_and_sample_derivatives(df, 'Volume', N=500,s=0.5e1)
    # fit_results_dev = {
    #     't_values': res['Development']['time'],
    #     'fits': res['Development']['fitted_values']
    # }
    # fit_results_reg = {
    #     't_values': res['Regeneration']['time'],
    #     'fits': res['Regeneration']['fitted_values']
    # }

    # fit_results_4850 = {
    #     't_values': res['4850cut']['time'],
    #     'fits': res['4850cut']['fitted_values']
    # }

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
    # p = figure(title=f"relative Volume growth rate",
    #            x_axis_label=r'hpf [t]',
    #            y_axis_label=r'$$\dot{V}/V \quad [1/h]$$',
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

    # fit_results_4850 = {
    #     't_values': res['4850cut']['time'],
    #     'fits': res['4850cut']['relative_derivative']
    # }
    # p = add_fit_to_plot_II(p, fit_results_dev, color='blue',label='Development')  
    # p = add_fit_to_plot_II(p, fit_results_reg, color='orange',label='Regeneration')  
    # p = add_fit_to_plot_II(p, fit_results_4850, color='black',label='48hpf 50%')  
    # p.xaxis.axis_label_text_font_size = "14pt"
    # p.yaxis.axis_label_text_font_size = "14pt"
    # p.xaxis.major_label_text_font_size = "12pt"
    # p.yaxis.major_label_text_font_size = "12pt"
    # show(p)
    res=fit_and_sample_derivatives(df, 'Volume', N=1000,s=0.5e1)
    fit_results_dev = {
        't_values': res['Development']['time'],
        'fits': res['Development']['derivative']
    }
    fit_results_reg = {
        't_values': res['Regeneration']['time'],
        'fits': res['Regeneration']['derivative']
    }
    plot_fit_with_uncertainty(
        fit_results_list=[fit_results_dev, fit_results_reg],
        colors=['blue', 'orange'],
        labels=['Development', 'Regeneration'],
        title='Volume Growth Rate',
        xlabel='t [hpf]',
        ylabel=r'$\dot{V} \quad [\mu m^3/h]$',
        ymin=0,
        xmin=48,
        dx=12,
        dy=10000,
    )

    fit_results_dev = {
        't_values': res['Development']['time'],
        'fits': res['Development']['relative_derivative']
    }
    fit_results_reg = {
        't_values': res['Regeneration']['time'],
        'fits': res['Regeneration']['relative_derivative']
    }
    plot_fit_with_uncertainty(
        fit_results_list=[fit_results_dev, fit_results_reg],
        colors=['blue', 'orange'],
        labels=['Development', 'Regeneration'],
        title='Relative Volume Growth Rate',
        xlabel='t [hpf]',
        ylabel=r'$\dot{V}/V \quad [1/h]$',
        ymin=0,
        xmin=48,
        dx=12,
        dy=0.02,
    )

    res=fit_and_sample_derivatives(df, 'Surface Area', N=1000,s=0.5e1)
    fit_results_dev = {
        't_values': res['Development']['time'],
        'fits': res['Development']['derivative']
    }
    fit_results_reg = {
        't_values': res['Regeneration']['time'],
        'fits': res['Regeneration']['derivative']
    }
    plot_fit_with_uncertainty(
        fit_results_list=[fit_results_dev, fit_results_reg],
        colors=['blue', 'orange'],
        labels=['Development', 'Regeneration'],
        title='Area Growth Rate',
        xlabel='t [hpf]',
        ylabel=r'$\dot{A} \quad [\mu m^2/h]$',
        ymin=0,
        xmin=48,
        dx=12,
        dy=1000,
    )

    fit_results_dev = {
        't_values': res['Development']['time'],
        'fits': res['Development']['relative_derivative']
    }
    fit_results_reg = {
        't_values': res['Regeneration']['time'],
        'fits': res['Regeneration']['relative_derivative']
    }
    plot_fit_with_uncertainty(
        fit_results_list=[fit_results_dev, fit_results_reg],
        colors=['blue', 'orange'],
        labels=['Development', 'Regeneration'],
        title='Relative Area Growth Rate',
        xlabel='t [hpf]',
        ylabel=r'$\dot{A}/A \quad [1/h]$',
        ymin=0,
        xmin=48,
        dx=12,
        dy=0.02,
    )

    #endregion

    # plot_single_timeseries(df, filter_col='condition', filter_value='Regeneration', y_col='Volume', style='violin', color='orange',width=None)
    # plot_double_timeseries(df, y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Tissue Volume $$10^6 \mu m^3$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='L PD', style='violin',y_scaling=1,y_name=r'$$L_{DP} in \mu m$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='L AP', style='violin',y_scaling=1,y_name=r'$$L_{AP} in \mu m$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='L DV', style='violin',y_scaling=1,y_name=r'$$L_{DV} in \mu m$$',test_significance=True,y0=0)
    
    
if __name__ == "__main__":
    main()