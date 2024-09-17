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
    df = pd.read_hdf(hdf5_file_path, key='data')


    df = df[df['experimentalist'] == 'Lucas']
    df = df.drop(columns=['ED_area', 'ED_PD', 'ED_AP'])
    df.rename(columns={
        'LM_folder': 'Mask Folder',
        'is control': 'condition',
        'total_area': 'Surface Area',
        'total_PD': 'L PD',
        'total_AP': 'L AP'
    }, inplace=True)
    df_Area = df[df['genotype'] == 'WT']
    df_Area['condition'] = df_Area['condition'].map({True: 'Development', False: 'Regeneration'})

    hdf5_file_path = os.path.join(Lucas_res, 'volumes.h5')
    df = pd.read_hdf(hdf5_file_path, key='data')

    df = df[df['experimentalist'] == 'Lucas']
    df.rename(columns={
        'Name': 'Mask Folder',
        'Is control': 'condition',
        'Time in hpf':'time in hpf',
        'Genotype':'genotype'
    }, inplace=True)
    df_Volume = df[df['genotype'] == 'WT']
    df_Volume['condition'] = df_Volume['condition'].map({True: 'Development', False: 'Regeneration'})

    pd.set_option('display.max_colwidth', None)  # Show full content of cells
    #pd.set_option('display.max_rows', None)      # Show all rows
    pd.set_option('display.max_columns', None)   # Show all columns
    pd.set_option('display.width', None)         # Avoid wrapping

    print(df_Volume)
    print(df_Area)

    df_Volume['Mask Folder Clean'] = df_Volume['Mask Folder'].str.replace('_vol$', '', regex=True)
    df_Area['Mask Folder Clean'] = df_Area['Mask Folder'].str.replace('_nuclei_LMcoord$', '', regex=True)

    df_merged = pd.merge(df_Volume, df_Area, on='Mask Folder Clean', suffixes=('', '_area'))
    df_merged = df_merged.drop(columns=['Mask Folder Clean','Mask Folder_area','experimentalist_area', 
                                    'condition_area',
                                    'time in hpf_area', 
                                    'genotype_area'])

    df_merged['L AP * L PD'] = df_merged['L AP'] * df_merged['L PD']
    df_merged['V / A'] = df_merged['Volume'] / df_merged['Surface Area']

    
    return df_merged


def main():
    df_Lucas=getData_Lucas()
    print(df_Lucas)
    
    df_SS=getData_SS()
    print(df_SS)

    

    plot_compare_timeseries(df_Lucas, df_SS, category='Development',y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,label_df1="Lucas", label_df2="S+S")
    plot_compare_timeseries(df_Lucas, df_SS, category='Regeneration',y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,label_df1="Lucas", label_df2="S+S")
    
    plot_compare_timeseries(df_Lucas, df_SS, category='Development',y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$',test_significance=True,label_df1="Lucas", label_df2="S+S")
    plot_compare_timeseries(df_Lucas, df_SS, category='Regeneration',y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$',test_significance=True,label_df1="Lucas", label_df2="S+S")

    plot_compare_timeseries(df_Lucas, df_SS, category='Development',y_col='V / A', style='violin',y_scaling=1.0,y_name=r'Mean thickness $$(\mu m)$$',test_significance=True,label_df1="Lucas", label_df2="S+S")
    plot_compare_timeseries(df_Lucas, df_SS, category='Regeneration',y_col='V / A', style='violin',y_scaling=1.0,y_name=r'Mean thickness $$(\mu m)$$',test_significance=True,label_df1="Lucas", label_df2="S+S")
    #plot_compare_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0,fit_results=fit)
    # plot_double_timeseries(df, y_col='Surface Area', style='violin',y_scaling=1e-4,y_name=r'Area $$(100 \mu m)^2$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='Volume', style='violin',y_scaling=1e-6,y_name=r'Volume $$(100 \mu m)^3$$',test_significance=True,y0=0)
    # plot_double_timeseries(df, y_col='V / A', style='violin',y_scaling=1.0,y_name=r'Mean thickness $$(\mu m)$$',test_significance=True,y0=0)



if __name__ == "__main__":
    main()