import os
import numpy as np
from zf_pf_diffeo.plot_movie import show_temporal_mesh_evolution,movie_temporal_hist_evolution
from zf_pf_diffeo.plot_static import plot_all_reference_meshes,plot_all_reference_data
from zf_pf_diffeo.pipeline import do_temporalHistInterpolation

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
    # raise
    print(hist_data['main_curvatures'])
    totalcc = np.array([np.median(cc) for cc in hist_data['main_curvatures']])
    
    mean_value = np.nanmean(totalcc)

    # Replace NaN values with the computed mean
    totalcc[np.isnan(totalcc)] = mean_value
    print(totalcc)
    return totalcc

if __name__ == "__main__":
    # Define folder paths
    proj_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/projected_surfaces"
    maps_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/Maps"
    temp_maps_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/Maps_temp"
    
    #plot_all_reference_meshes(maps_dir, scale_unit="µm")
    plot_all_reference_data(maps_dir, data_to_value_function_Volume, scale_unit="µm",separate_windows=False)
    raise
    
    show_temporal_mesh_evolution(os.path.join(temp_maps_dir,"Vinita_WT_Development"))
    
    do_temporalHistInterpolation(proj_dir,temp_maps_dir, "time in hpf", ["experimentalist","genotype","condition"], {"BRE": data_to_value_function},surface_key="projected_data",surface_file_key="Projected Surface file name")

    movie_temporal_hist_evolution(os.path.join(temp_maps_dir,"Vinita_WT_Development"),save_path="/home/max/Downloads/VINITA/dev_BRE.mp4",scale_unit="µm")
    movie_temporal_hist_evolution(os.path.join(temp_maps_dir,"Vinita_WT_Regeneration"),save_path="/home/max/Downloads/VINITA/reg_BRE.mp4",scale_unit="µm")