import pandas as pd
import os
import numpy as np

from config import *

#from plotMatPlotHelper import 
from plotBokehHelper import plot_single_timeseries,plot_double_timeseries,plot_scatter,plot_explanation,plot_compare_timeseries

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
    print(df)
    # 1. Ensure all rows have 'Lucas' as the experimentalist
    df = df[df['experimentalist'] == 'Lucas']

    # 2. Remove unwanted columns
    df = df.drop(columns=['ED_area', 'ED_PD', 'ED_AP'])

    # 3. Rename columns to match df_SS
    df.rename(columns={
        'LM_folder': 'Mask Folder',
        'is control': 'condition',
        'total_area': 'Surface Area',
        'total_PD': 'L PD',
        'total_AP': 'L AP'
    }, inplace=True)

    # 4. Filter only rows where 'genotype' is 'WT'
    df = df[df['genotype'] == 'WT']

    # 5. Map 'condition' from True/False to 'Development'/'Regeneration'
    df['condition'] = df['condition'].map({True: 'Development', False: 'Regeneration'})

    # 6. Calculate additional columns as in df_SS
    df['L AP * L PD'] = df['L AP'] * df['L PD']
    return df


def main():
    df_Lucas=getData_Lucas()
    print(df_Lucas)
    
    df_SS=getData_SS()
    print(df_SS)

    

    plot_compare_timeseries(df_Lucas, df_SS, category='Development', y_col='Surface Area', style='box')

    #plot_compare_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0,fit_results=fit)
    # plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='V / A', style='violin',y_scaling=1.0,y_name=r'Mean thickness $$(\mu m)$$',test_significance=True,y0=0)



if __name__ == "__main__":
    main()