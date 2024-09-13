import pandas as pd
import os
import numpy as np
import re
import pyvista as pv

from config import *
from IO import *

#from plotMatPlotHelper import 
from plotBokehHelper import plot_histograms_bokeh,plot_scatter_quantities,plot_density_hexbin

def extract_category_and_number(input_string):
    # Define the regex pattern to match the required format
    pattern = r'([a-zA-Z]+)_(\d+)'
    
    # Use the re.match method to apply the regex pattern to the input string
    match = re.match(pattern, input_string)
    
    if match:
        # Extract the category and number from the match object
        category = match.group(1)
        number = int(match.group(2))
        return category, number
    else:
        # If the input string doesn't match the pattern, return None
        return None

def getData():
    relevant_conditions=['time in hpf','condition']
    relevant_conditions.sort()
    combined_string = '_'.join(relevant_conditions)
    Condition_path_dir=os.path.join(Hist_path,combined_string)
    print(Condition_path_dir)
    hist_folder_list=[folder for folder in os.listdir(Condition_path_dir) if os.path.isdir(os.path.join(Condition_path_dir, folder))]
    times=[]
    categories=[]
    meshes=[]
    for hist_folder in hist_folder_list:
        hist_dir_path=os.path.join(Condition_path_dir,hist_folder)

        MetaData=get_JSON(hist_dir_path)
        
        if not 'Hist_MetaData' in MetaData:
            continue
        MetaData=MetaData['Hist_MetaData']
        print(hist_folder)
        
        surface_file=MetaData['Surface file']
        Mesh_file_path=os.path.join(hist_dir_path,surface_file)
        mesh=pv.read(Mesh_file_path)

        mesh.point_data['mean2-gauss']=mesh.point_data['mean_curvature_avg']*mesh.point_data['mean_curvature_avg']-mesh.point_data['gauss_curvature_avg']
            
        #data_file=MetaData['Data file']
        #data_file_path=os.path.join(hist_dir_path,data_file)
        #hist_data=np.load(data_file_path+'.npy',allow_pickle=True)
        
        category,time=extract_category_and_number(hist_folder)
        
        times.append(time)
        categories.append(category)
        meshes.append(mesh)
 
    return times,categories,meshes

# Organize meshes by category and time
    # dev_meshes = [(time, mesh) for time, category, mesh in zip(times, categories, meshs) if category == 'Development']
    # reg_meshes = [(time, mesh) for time, category, mesh in zip(times, categories, meshs) if category == 'Regeneration']

    # dev_meshes.sort()
    # reg_meshes.sort()


def main():
    #TODO scatter check gauss_avg vs gaus from curvature_mean
    times,categories,meshes=getData()
    print(times,categories)
    #plot_histograms_bokeh(times, categories, meshes,quantities_to_plot=['thickness_avg'])
    #plot_histograms_bokeh(times, categories, meshes,quantities_to_plot=['thickness_avg'],combinations={'combine_times_per_category': True})

    #plot_histograms_bokeh(times, categories, meshes,quantities_to_plot=['gauss_curvature_avg'])
    #plot_histograms_bokeh(times, categories, meshes,quantities_to_plot=['thickness_avg'],combinations={'combine_times_per_category': True})

    # plot_scatter_quantities(times, categories, meshes,
    #                         x_quantity='mean_curvature_avg',
    #                         y_quantity='gauss_curvature_avg',
    #                         mode='time')
    
    plot_density_hexbin(times, categories, meshes,
                        x_quantity='mean_curvature_avg',
                        y_quantity='gauss_curvature_avg',rel_aspect_scale=0.5,rel_size=0.002)



if __name__ == "__main__":
    main()