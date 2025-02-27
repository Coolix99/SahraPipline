import os
import numpy as np
from zf_pf_diffeo.plot_movie import show_temporal_mesh_evolution,movie_temporal_hist_evolution
from zf_pf_diffeo.plot_static import plot_all_reference_meshes,plot_all_reference_data
from zf_pf_diffeo.pipeline import do_temporalHistInterpolation

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

def data_to_value_function(hist_data):
    #print("Keys in .npz file:", hist_data.files)
    
    totalcc = np.array([np.median(cc) for cc in hist_data['BRE_max_intensity']])
    
    mean_value = np.nanmean(totalcc)

    # Replace NaN values with the computed mean
    totalcc[np.isnan(totalcc)] = mean_value
    #print(totalcc)
    return totalcc

def data_to_value_function_Volume(hist_data):
    # print("Keys in .npz file:", hist_data.files)
    # #print(hist_data['Volume_sum'])
    # raise
    
    total_sum = np.array([np.sum(cc) for cc in hist_data['Volume_sum']])
    total_count = np.array([np.sum(cc) for cc in hist_data['Volume_count']])

    # Avoid division by zero
    avg = np.zeros_like(total_sum, dtype=float)
    valid = total_count > 0  # Mask where count is greater than zero
    avg[valid] = total_sum[valid] / total_count[valid]
    return avg

def data_to_value_function_SurfaceArea(hist_data):
    # print("Keys in .npz file:", hist_data.files)
    # #print(hist_data['Volume_sum'])
    # raise
    
    total_sum = np.array([np.sum(cc) for cc in hist_data['Surface Area_sum']])
    total_count = np.array([np.sum(cc) for cc in hist_data['Surface Area_count']])

    # Avoid division by zero
    avg = np.zeros_like(total_sum, dtype=float)
    valid = total_count > 0  # Mask where count is greater than zero
    avg[valid] = total_sum[valid] / total_count[valid]

    return avg

def data_to_value_function_Solidity(hist_data):
    # print("Keys in .npz file:", hist_data.files)
    # print(hist_data['Solidity_sum'])
    # raise
    
    total_sum = np.array([np.sum(cc) for cc in hist_data['Solidity_sum']])
    total_count = np.array([np.sum(cc) for cc in hist_data['Solidity_count']])

    # Avoid division by zero
    avg = np.zeros_like(total_sum, dtype=float)
    valid = total_count > 0  # Mask where count is greater than zero
    avg[valid] = total_sum[valid] / total_count[valid]
    avg = np.where(np.isfinite(avg), avg, 0)
    return avg

def data_to_value_function_Elongation(hist_data):
    # print("Keys in .npz file:", hist_data.files)
    # #print(hist_data['Volume_sum'])
    # raise
    
    total_sum = np.array([np.sum(cc) for cc in hist_data['2I1/I2+I3_sum']])
    total_count = np.array([np.sum(cc) for cc in hist_data['2I1/I2+I3_count']])

    # Avoid division by zero
    avg = np.zeros_like(total_sum, dtype=float)
    valid = total_count > 0  # Mask where count is greater than zero
    avg[valid] = total_sum[valid] / total_count[valid]
    avg = np.where(np.isfinite(avg), avg, 0)

    avg[avg<0]=0
    avg[avg>10]=10

    return avg

if __name__ == "__main__":
    # Define folder paths
    # proj_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/projected_surfaces"
    # maps_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/Maps"
    # temp_maps_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/Maps_temp"
    proj_dir = os.path.join(Output_path,"morphoMaps","projected_surfaces")
    maps_dir = os.path.join(Output_path,"morphoMaps","Maps")
    temp_maps_dir = os.path.join(Output_path,"morphoMaps","Maps_temp")

    plot_all_reference_meshes(maps_dir, scale_unit="µm")
    plot_all_reference_data(maps_dir, data_to_value_function_Volume, scale_unit="µm",separate_windows=False,vmin=100,vmax=400)
    plot_all_reference_data(maps_dir, data_to_value_function_SurfaceArea, scale_unit="µm",separate_windows=False,vmin=100)
    plot_all_reference_data(maps_dir, data_to_value_function_Solidity, scale_unit="µm",separate_windows=False)
    plot_all_reference_data(maps_dir, data_to_value_function_Elongation, scale_unit="µm",separate_windows=False,vmin=1.0,vmax=10.0)
    plot_all_reference_data(maps_dir, data_to_value_function_Elongation, scale_unit="µm",separate_windows=False,vmin=1.0,vmax=5.0)
    plot_all_reference_data(maps_dir, data_to_value_function_Elongation, scale_unit="µm",separate_windows=False,vmin=1.0,vmax=3.0)
    raise
    
    show_temporal_mesh_evolution(os.path.join(temp_maps_dir,"WT_Development"))
    show_temporal_mesh_evolution(os.path.join(temp_maps_dir,"WT_Regeneration"))
    
    do_temporalHistInterpolation(proj_dir,temp_maps_dir, "time in hpf", ["genotype","condition"], {"vol": data_to_value_function_Volume,
                                                                                                   "area": data_to_value_function_SurfaceArea,
                                                                                                   "sol": data_to_value_function_Solidity,
                                                                                                   "el":data_to_value_function_Elongation},surface_key="projected_data",surface_file_key="Projected Surface file name")

    movie_temporal_hist_evolution(os.path.join(temp_maps_dir,"WT_Development"),scale_unit="µm")#save_path="/home/max/Downloads/VINITA/dev_BRE.mp4"
    movie_temporal_hist_evolution(os.path.join(temp_maps_dir,"WT_Regeneration"),scale_unit="µm")#save_path="/home/max/Downloads/VINITA/reg_BRE.mp4"