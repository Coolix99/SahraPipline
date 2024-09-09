import pandas as pd
import os

from config import *

#from plotMatPlotHelper import plot_single_timeseries,plot_double_timeseries
from plotBokehHelper import plot_single_timeseries,plot_double_timeseries

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

    #simple_plot(df, filter_col='condition', filter_value='Regeneration', y_col='Volume') #just debuggin

    #plot_single_timeseries(df, filter_col='condition', filter_value='Regeneration', y_col='Volume', style='violin', color='orange',width=None)
    #plot_double_timeseries(df, y_col='Volume', style='violin')
    #plot_double_timeseries(df, y_col='Surface Area', style='box')
    plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$')
    plot_double_timeseries(df, y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$')



if __name__ == "__main__":
    main()