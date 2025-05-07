import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *


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



def fit_and_sample_derivatives(df: pd.DataFrame, column: str, N: int = 200,s: float = 0):
    results = {}
    grouped = df.groupby('condition')

    for condition, group in grouped:
        if  condition == '4850cut' or condition == '7230cut':
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

    res=fit_and_sample_derivatives(df, 'Volume', N=1000,s=0.4e1)
    fit_results_dev = {
        't_values': res['Development']['time'],
        'fits': res['Development']['fitted_values']
    }
    fit_results_reg = {
        't_values': res['Regeneration']['time'],
        'fits': res['Regeneration']['fitted_values']
    }
    plot_fit_with_uncertainty(
        fit_results_list=[fit_results_dev, fit_results_reg],
        colors=['blue', 'orange'],
        labels=['Development', 'Regeneration'],
        title='Volume ',
        xlabel='t [hpf]',
        ylabel=r'$\dot{V} \quad [\mu m^3]$',
        # ymin=0,
        # xmin=48,
        # dx=12,
        # dy=10000,
    )

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


    
    
if __name__ == "__main__":
    main()