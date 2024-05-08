import pandas as pd
import networkx as nx
import pyvista as pv
from typing import List
import git
import napari
from simple_file_checksum import get_checksum
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from config import *
from IO import *

def filter_outliers(data, n_std=3):
    mean = np.mean(data)
    std = np.std(data)
    filtered = np.where(np.abs(data - mean) > n_std * std, np.nan, data)
    return filtered

def plot3d_napari(mesh:pv.PolyData):
    print(mesh.point_data)
    viewer = napari.Viewer(ndisplay=3)
    vertices = mesh.points
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]

    for key,nstd in zip(['thickness_avg', 'avg_curvature_avg', 'gauss_curvature_avg'],[5,4,4]):
        data = mesh.point_data[key]
        filtered_data = filter_outliers(data,nstd)
        viewer.add_surface((vertices, faces, filtered_data), name=key, colormap='turbo',shading='none')

    napari.run()

def plot2d(mesh):
    # Extracting vertex coordinates and face indices
    coord_1 = mesh.point_data['coord_1']
    coord_2 = mesh.point_data['coord_2']
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]
    triangulation = tri.Triangulation(coord_1, coord_2, faces)
    
    for key,nstd in zip(['thickness_avg', 'avg_curvature_avg', 'gauss_curvature_avg'],[5,4,4]):
        data = mesh.point_data[key]
        filtered_data = filter_outliers(data,nstd)
        
    
        # Plotting the triangles with curvature data as color
        plt.figure(figsize=(10, 8))
        coloring = plt.tripcolor(triangulation, filtered_data, shading='flat', cmap='viridis')
        plt.colorbar(coloring, label=key)
        plt.xlabel('Coord 1')
        plt.ylabel('Coord 2')
        plt.grid(True)
        plt.show()



def plotHistogramms(relevant_conditions):
    relevant_conditions.sort()
    combined_string = '_'.join(relevant_conditions)
    Condition_path_dir=os.path.join(Hist_path,combined_string)
    print(Condition_path_dir)
    
    hist_folder_list=[folder for folder in os.listdir(Condition_path_dir) if os.path.isdir(os.path.join(Condition_path_dir, folder))]
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
        
        data_file=MetaData['Data file']
        data_file_path=os.path.join(hist_dir_path,data_file)
        hist_data=np.load(data_file_path+'.npy',allow_pickle=True)
        
        plot3d_napari(mesh)

        plot2d(mesh)

        
    

def main():  
    plotHistogramms(['time in hpf','condition'])
    plotHistogramms(['condition'])
    plotHistogramms(['time in hpf'])
    return
    
    pass

if __name__ == "__main__":
    main()