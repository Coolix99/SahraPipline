import pyvista as pv
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
pd.set_option('display.max_colwidth', None)
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance_matrix
from scipy.sparse import find 

from config import *
from IO import *

def calculate_mesh_features(mesh:pv.PolyData):
    features = {}
    #print(mesh.point_data)
    features['surface_area'] = mesh.area
    # features['average_curvature'] = mesh.point_data['avg_curvature'].mean()
    # features['gauss_curvature'] = mesh.point_data['gauss_curvature'].mean()
    # features['max average_curvature'] = np.abs(mesh.point_data['avg_curvature']).max()
    # features['max gauss_curvature'] = np.abs(mesh.point_data['gauss_curvature']).max()
    features['L1'] = np.max(mesh.point_data['coord_1'])-np.min(mesh.point_data['coord_1'])
    features['L2'] = np.max(mesh.point_data['coord_2'])-np.min(mesh.point_data['coord_2'])
    features['aspect ratio'] = features['L1']/features['L2']


    return features

def get_primitive_Graph(df):
    from scipy.spatial import KDTree
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.transform import factor_mark, linear_cmap
    from bokeh.palettes import Viridis256
    if len(df) == 1:
        return None

    meshes = []
    for row in df.itertuples():
        meshes.append(pv.read(os.path.join(FlatFin_path, row.folder_name, row.file_name)))

    # Calculate features
    features_list = [calculate_mesh_features(mesh) for mesh in meshes]
    feature_df = pd.DataFrame(features_list)[['surface_area', 'aspect ratio']]
    
    # Normalize the features
    scaler = StandardScaler()
    feature_df_normalized = scaler.fit_transform(feature_df)
    
    # Add normalized features back to the dataframe
    df['x'] = feature_df_normalized[:, 0]
    df['y'] = feature_df_normalized[:, 1]
    
    # Create a ColumnDataSource from the dataframe
    source = ColumnDataSource(df)
    
    # Define unique markers for different conditions
    conditions = df['condition'].unique()
    markers = ['circle', 'triangle', 'square', 'diamond', 'inverted_triangle']
    
    # Create the Bokeh figure
    p = figure(title="Feature Scatter Plot", tools="pan,wheel_zoom,box_zoom,reset")
    
    # Add scatter plot with different markers and colors
    p.scatter('x', 'y', source=source,
              fill_alpha=0.6, size=10,
              marker=factor_mark('condition', markers, conditions),
              color=linear_cmap('time in hpf', Viridis256, df['time in hpf'].min(), df['time in hpf'].max()))
    
    # Add hover tool to show the file_name
    hover = HoverTool()
    hover.tooltips = [
        ("File Name", "@file_name"),
        ("Condition", "@condition"),
        ("Time in hpf", "@{time in hpf}"),
        ("Surface Area", "@x"),
        ("Aspect Ratio", "@y"),
    ]
    p.add_tools(hover)

    points = np.column_stack((df['x'], df['y']))
    dist_matrix = distance_matrix(points, points)
    mst = minimum_spanning_tree(dist_matrix).toarray()

    # Build the graph structure
    graph = {row.folder_name: [] for row in df.itertuples()}
    edges = []

    for i in range(len(points)):
        for j in range(len(points)):
            if mst[i, j] > 0:
                graph[df.iloc[i]['folder_name']].append(df.iloc[j]['folder_name'])
                edges.append((i, j))
    
    # Draw the edges as lines between points
    for edge in edges:
        p.line([points[edge[0], 0], points[edge[1], 0]], 
               [points[edge[0], 1], points[edge[1], 1]], 
               line_width=2, color='black')
    
    show(p)
    print(graph)
    # Return the graph in a dictionary format
    return graph

def make_primitive_Graph():
    FlatFin_folder_list=os.listdir(FlatFin_path)
    FlatFin_folder_list = [item for item in FlatFin_folder_list if os.path.isdir(os.path.join(FlatFin_path, item))]
    data_list = []
    for FlatFin_folder in FlatFin_folder_list:
        print(FlatFin_folder)
        FlatFin_dir_path=os.path.join(FlatFin_path,FlatFin_folder)
        MetaData=get_JSON(FlatFin_dir_path)
        if not 'Thickness_MetaData' in MetaData:
            continue
        MetaData=MetaData['Thickness_MetaData']

        data_list.append({
            'folder_name':FlatFin_folder,
            'file_name': MetaData['Surface file'],
            'condition': MetaData['condition'],
            'time in hpf': MetaData['time in hpf'],
            'genotype': MetaData['genotype'],
            'experimentalist': MetaData['experimentalist']
        })
    df = pd.DataFrame(data_list)

    print(df)
    
    make_path(primitive_Graph_path)

    
    get_primitive_Graph(df)
    return
    
       

def main():
    make_primitive_Graph()

    return

if __name__ == "__main__":
    main()