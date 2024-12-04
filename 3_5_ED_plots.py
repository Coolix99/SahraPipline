import os
import pandas as pd
from plotBokehHelper import plot_double_timeseries
from plotHelper import plot_scatter_corner
from bokeh.io import show

from config import *
from IO import *

def collect_dfs():
     # Initialize an empty list to collect all dataframes
    all_dfs = []

    # Iterate through folders
    EDprops_folder_list = [item for item in os.listdir(ED_cell_props_path) if os.path.isdir(os.path.join(ED_cell_props_path, item))]
    for EDprop_folder in EDprops_folder_list:
        print(EDprop_folder)
        EDprop_folder_path = os.path.join(ED_cell_props_path, EDprop_folder)

        EDpropMetaData = get_JSON(EDprop_folder_path)
        if not EDpropMetaData:
            print('No EDprops found')
            continue

        # Load the dataframe
        df_prop = pd.read_hdf(os.path.join(EDprop_folder_path, EDpropMetaData['MetaData_EDcell_props']['EDcells file']), key='data')

        # Add metadata as new columns
        df_prop['time in hpf'] = EDpropMetaData['MetaData_EDcell_props']['time in hpf']
        df_prop['condition'] = EDpropMetaData['MetaData_EDcell_props']['condition']

        # Append the dataframe to the list
        all_dfs.append(df_prop)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

def main():
    # merged_df=collect_dfs()
    # merged_df.to_hdf('data.h5', key='df', mode='w')
    merged_df = pd.read_hdf('data.h5', key='df')
    print(merged_df.columns)
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
    merged_df = merged_df.dropna()

    merged_df['2I1/I2+I3']=2*merged_df['inertia_tensor_eigvals 1']/(merged_df['inertia_tensor_eigvals 2']+merged_df['inertia_tensor_eigvals 3'])
    merged_df['I1+I2/2I3']=(merged_df['inertia_tensor_eigvals 1']+merged_df['inertia_tensor_eigvals 2'])/(2*merged_df['inertia_tensor_eigvals 3'])
    
    # plot_double_timeseries(merged_df, y_col='Volume', style='violin',y_scaling=1,y_name=r'Cell Volume $$\mu m^3$$',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='Surface Area', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='Solidity', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='Sphericity', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='2I1/I2+I3', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='I1+I2/2I3', style='violin',test_significance=True,y0=0,show_scatter=False)

    color_dict = {'Regeneration': 'orange',
                 'Development': 'blue', 
                 }
    marker_dict = {'Development': 'circle', 'Regeneration': 'triangle', 'Smoc12': 'square'}
    corner_plot = plot_scatter_corner(df=merged_df, parameters=['Volume','Surface Area','Solidity', 'Sphericity','2I1/I2+I3', 'I1+I2/2I3'], color_col='time in hpf',color_dict=color_dict,marker_col='condition',marker_dict=marker_dict)
    show(corner_plot)

if __name__ == "__main__":
    main()
