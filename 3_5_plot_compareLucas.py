import pandas as pd
import os
import numpy as np

from config import *

#from plotMatPlotHelper import 
from plotBokehHelper import plot_single_timeseries,plot_double_timeseries,plot_scatter,plot_explanation

def getData_SS():
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

def getData_Lucas():
    hdf5_file_path = os.path.join(Lucas_res, 'SurfaceSizes.h5')

    # Load the DataFrame from the HDF5 file
    df = pd.read_hdf(hdf5_file_path, key='data')

def main():
    df_Lucas=getData_Lucas()
    print(df_Lucas)
    return
    df_SS=getData_SS()
    print(df_SS)

    

    #Checks
    # plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='category',show_fit=True,show_div=True)
    # plot_scatter(df, x_col='Integrated Thickness', y_col='Volume', mode='time',show_fit=True,show_div=True)

    # plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='category',show_fit=True,show_div=True)
    # plot_scatter(df, x_col='L AP * L PD', y_col='Surface Area', mode='time',show_fit=True,show_div=True)

    # plot_scatter(df, x_col='L AP', y_col='L PD', mode='category',show_fit=True,show_div=True)
    # plot_scatter(df, x_col='L AP', y_col='L PD', mode='time',show_fit=True,show_div=True)

    # #simple_plot(df, filter_col='condition', filter_value='Regeneration', y_col='Volume') #just debuggin

    # plot_single_timeseries(df, filter_col='condition', filter_value='Regeneration', y_col='Volume', style='violin', color='orange',width=None)
    # plot_double_timeseries(df, y_col='Volume', style='violin')
    # plot_double_timeseries(df, y_col='Surface Area', style='box')
    # fit={
    #     't_values':t_values,
    #     'Development': results['A_Development_noisy'],
    #     'Regeneration': results['A_Regeneration_noisy']
    # }
    # plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0,fit_results=fit)
    # plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='V / A', style='violin',y_scaling=1.0,y_name=r'Mean thickness $$(\mu m)$$',test_significance=True,y0=0)



if __name__ == "__main__":
    main()