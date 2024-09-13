import pandas as pd
import pyvista as pv
import napari
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import re

from config import *
from IO import *

# def filter_outliers(data, n_std=3):
#     mean = np.mean(data)
#     std = np.std(data)
#     filtered = np.where(np.abs(data - mean) > n_std * std, np.nan, data)
#     return filtered

# def plot3d_napari(mesh:pv.PolyData):
#     print(mesh.point_data)
#     viewer = napari.Viewer(ndisplay=3)
#     vertices = mesh.points
#     faces = mesh.faces.reshape((-1, 4))[:, 1:4]

#     for key,nstd in zip(['thickness_avg', 'mean_curvature_avg', 'gauss_curvature_avg'],[5,4,4]):
#         data = mesh.point_data[key]
#         filtered_data = filter_outliers(data,nstd)
#         viewer.add_surface((vertices, faces, filtered_data), name=key, colormap='turbo',shading='none')

#     napari.run()

# def plot2d(mesh):
#     # Extracting vertex coordinates and face indices
#     coord_1 = mesh.point_data['coord_1']
#     coord_2 = mesh.point_data['coord_2']
#     faces = mesh.faces.reshape((-1, 4))[:, 1:4]
#     triangulation = tri.Triangulation(coord_1, coord_2, faces)
    
#     for key,nstd in zip(['thickness_avg', 'mean_curvature_avg', 'gauss_curvature_avg'],[5,4,4]):
#         data = mesh.point_data[key]
#         filtered_data = filter_outliers(data,nstd)
        
    
#         # Plotting the triangles with curvature data as color
#         plt.figure(figsize=(10, 8))
#         coloring = plt.tripcolor(triangulation, filtered_data, shading='flat', cmap='viridis')
#         plt.colorbar(coloring, label=key)
#         plt.xlabel('Coord 1')
#         plt.ylabel('Coord 2')
#         plt.grid(True)
#         plt.show()

# def plotHistogramms(relevant_conditions):
#     relevant_conditions.sort()
#     combined_string = '_'.join(relevant_conditions)
#     Condition_path_dir=os.path.join(Hist_path,combined_string)
#     print(Condition_path_dir)
    
#     hist_folder_list=[folder for folder in os.listdir(Condition_path_dir) if os.path.isdir(os.path.join(Condition_path_dir, folder))]
#     for hist_folder in hist_folder_list:
#         hist_dir_path=os.path.join(Condition_path_dir,hist_folder)

#         MetaData=get_JSON(hist_dir_path)
        
#         if not 'Hist_MetaData' in MetaData:
#             continue
#         MetaData=MetaData['Hist_MetaData']
#         print(hist_folder)
        
#         surface_file=MetaData['Surface file']
#         Mesh_file_path=os.path.join(hist_dir_path,surface_file)
#         mesh=pv.read(Mesh_file_path)
        
#         data_file=MetaData['Data file']
#         data_file_path=os.path.join(hist_dir_path,data_file)
#         hist_data=np.load(data_file_path+'.npy',allow_pickle=True)
        
#         #plot3d_napari(mesh)

#         plot2d(mesh)

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

# def plot_mesh_2d(ax, mesh, key, nstd, vmin, vmax):
#     coord_1 = mesh.point_data['coord_1']
#     coord_2 = mesh.point_data['coord_2']
#     faces = mesh.faces.reshape((-1, 4))[:, 1:4]
#     triangulation = tri.Triangulation(coord_1, coord_2, faces)
    
#     data = mesh.point_data[key]
#     filtered_data = filter_outliers(data, nstd)
    
#     coloring = ax.tripcolor(triangulation, filtered_data, shading='flat', cmap='viridis', vmin=vmin, vmax=vmax)
#     ax.set_xlabel('Coord 1')
#     ax.set_ylabel('Coord 2')
#     ax.set_aspect('equal')
#     ax.grid(True)
#     return coloring

def getData():
    relevant_conditions=['time in hpf','condition']
    relevant_conditions.sort()
    combined_string = '_'.join(relevant_conditions)
    Condition_path_dir=os.path.join(Hist_path,combined_string)
    print(Condition_path_dir)
    hist_folder_list=[folder for folder in os.listdir(Condition_path_dir) if os.path.isdir(os.path.join(Condition_path_dir, folder))]
    times=[]
    categories=[]
    meshs=[]
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
        meshs.append(mesh)
    return times,categories,meshs

# def plotCompare2d():
#     times, categories, meshs=getData()
#     # Organize meshes by category and time
#     dev_meshes = [(time, mesh) for time, category, mesh in zip(times, categories, meshs) if category == 'Development']
#     reg_meshes = [(time, mesh) for time, category, mesh in zip(times, categories, meshs) if category == 'Regeneration']

#     dev_meshes.sort()
#     reg_meshes.sort()
    
#     keys = ['thickness_avg', 'mean_curvature_avg', 'gauss_curvature_avg', 'mean2-gauss']
#     nstds = [5, 4, 4,4]
    
#     global_min_max = {}
#     for key in keys:
#         all_values = np.concatenate([mesh.point_data[key] for mesh in meshs])
#         global_min_max[key] = (all_values.min(), all_values.max())
#     print(global_min_max)
#     global_min_max['mean_curvature_avg']=(-0.01, 0.01)
#     global_min_max['gauss_curvature_avg']=(-0.0001, 0.0001)
#     global_min_max['mean2-gauss'] = (-0.00003, 0.00003)
#     from matplotlib.gridspec import GridSpec

#     for key, nstd in zip(keys, nstds):
#         print(key)
#         vmin, vmax = global_min_max[key]
#         fig = plt.figure(figsize=(25, 8))
#         gs = GridSpec(2, max(len(dev_meshes), len(reg_meshes)), figure=fig)
        
#         axs = []
#         for i in range(max(len(dev_meshes), len(reg_meshes))):
#             if i < len(dev_meshes):
#                 ax = fig.add_subplot(gs[0, i])
#                 coloring = plot_mesh_2d(ax, dev_meshes[i][1], key, nstd, vmin, vmax)
#                 ax.set_title(f'dev_{dev_meshes[i][0]}')
#                 axs.append(ax)
#             if i < len(reg_meshes):
#                 ax = fig.add_subplot(gs[1, i])
#                 coloring = plot_mesh_2d(ax, reg_meshes[i][1], key, nstd, vmin, vmax)
#                 ax.set_title(f'reg_{reg_meshes[i][0]}')
#                 axs.append(ax)
        
#         plt.subplots_adjust(bottom=0.2)
        
#         # Add colorbar in the margin at the bottom
#         cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])  # [left, bottom, width, height]
#         cbar = fig.colorbar(coloring, cax=cbar_ax, orientation='horizontal')
#         cbar.set_label(key)
        
#         plt.tight_layout(rect=[0, 0.2, 1, 1])
#         plt.show()

from plotMatGrid import plotHistogramsSequentially,plotHistogramsComparison

def main():
    # Load data
    times, categories, meshes = getData()
    print(meshes[0].point_data)
    return
    # Specify the keys (metrics) to plot
    keys = ['thickness_avg', 'mean_curvature_avg', 'gauss_curvature_avg', 'mean2-gauss']
    nstds = [5, 4, 4, 4]  # Number of standard deviations for filtering outliers
    
    # Sequential histogram plotting (one after the other)
    plotHistogramsSequentially(times, categories, meshes, keys, nstds)
    
    # Comparison histogram plotting (all histograms at once)
    plotHistogramsComparison(times, categories, meshes, keys, nstds)

if __name__ == "__main__":
    main()