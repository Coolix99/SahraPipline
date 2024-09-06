import pyvista as pv
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from bokeh.plotting import figure, show, curdoc
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS
from bokeh.transform import factor_mark, linear_cmap
from bokeh.palettes import Viridis256
import matplotlib.pyplot as plt

# Importing configurations and custom I/O
from config import *
from IO import *

def calculate_mesh_features(mesh: pv.PolyData):
    features = {}
    features['surface_area'] = mesh.area
    features['L1'] = np.max(mesh.point_data['coord_1']) - np.min(mesh.point_data['coord_1'])
    features['L2'] = np.max(mesh.point_data['coord_2']) - np.min(mesh.point_data['coord_2'])
    features['aspect_ratio'] = features['L1'] / features['L2']
    return features

def update_graph(new_edges, graph, points, p):
    for edge in new_edges:
        i, j = edge
        p.line([points[i][0], points[j][0]], 
               [points[i][1], points[j][1]], 
               line_width=2, color='black')

def add_edges_to_graph(graph, dist_matrix, min_connections=3):
    node_connections = {node: len(neighbors) for node, neighbors in graph.items()}
    
    for i in range(len(dist_matrix)):
        while node_connections[i] < min_connections:
            potential_connections = np.argsort(dist_matrix[i])
            for j in potential_connections:
                if j != i and j not in graph[i]:
                    graph[i].append(j)
                    graph[j].append(i)
                    node_connections[i] += 1
                    node_connections[j] += 1
                    break
    return graph


def on_click(event, points, graph, lines):
    if event.inaxes is not None:
        x, y = event.xdata, event.ydata
        distances = np.hypot(points[:, 0] - x, points[:, 1] - y)
        selected_point = np.argmin(distances)
        
        if not hasattr(on_click, 'selected'):
            on_click.selected = []
        
        on_click.selected.append(selected_point)
        
        if len(on_click.selected) == 2:
            i, j = on_click.selected
            if j not in graph[i]:
                graph[i].append(j)
                graph[j].append(i)
                line = plt.Line2D([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color='black')
                lines.append(line)
                event.inaxes.add_line(line)
                plt.draw()
            on_click.selected = []

def plot_graph(points, graph):
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=100)

    lines = []
    for i, neighbors in graph.items():
        for j in neighbors:
            if i < j:  # To avoid drawing the same edge twice
                line = plt.Line2D([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color='black')
                lines.append(line)
                ax.add_line(line)
    
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, points, graph, lines))
    plt.show()

def get_primitive_Graph(df):
    if len(df) == 1:
        return None

    meshes = []
    features_list=[]
    for row in df.itertuples():
        meshes.append(pv.read(os.path.join(FlatFin_path, row.folder_name, row.file_name)))
        features_list.append(calculate_mesh_features(meshes[-1]))
        features_list[-1]['folder_name']=row.folder_name
    feature_df = pd.DataFrame(features_list)[['surface_area', 'aspect_ratio','folder_name']]
    
    scaler = StandardScaler()
    feature_df_normalized = scaler.fit_transform(feature_df[['surface_area', 'aspect_ratio']])
    
    df['x'] = feature_df_normalized[:, 0]
    df['y'] = feature_df_normalized[:, 1]
    

    points = np.column_stack((df['x'], df['y']))
    dist_matrix = distance_matrix(points, points)
    mst = minimum_spanning_tree(dist_matrix).toarray()
    
    graph = {i: [] for i in range(len(points))}
   
    for i in range(len(points)):
        for j in range(len(points)):
            if mst[i, j] > 0:
                graph[i].append(j)

    graph = add_edges_to_graph(graph, dist_matrix, min_connections=3)

    source = ColumnDataSource(df)
    conditions = df['condition'].unique()
    markers = ['circle', 'triangle', 'square', 'diamond', 'inverted_triangle']
    p = figure(title="Feature Scatter Plot", tools="pan,wheel_zoom,box_zoom,reset,tap")
    
    p.scatter('x', 'y', source=source,
              fill_alpha=0.6, size=10,
              marker=factor_mark('condition', markers, conditions),
              color=linear_cmap('time in hpf', Viridis256, df['time in hpf'].min(), df['time in hpf'].max()))

    hover = HoverTool()
    hover.tooltips = [
        ("File Name", "@file_name"),
        ("Condition", "@condition"),
        ("Time in hpf", "@{time in hpf}"),
        ("Surface Area", "@x"),
        ("Aspect Ratio", "@y"),
    ]
    p.add_tools(hover)

    edges = []
    for i in range(len(points)):
        for j in graph[i]:
            if i < j:  # To avoid duplicate edges
                edges.append((i, j))
    for edge in edges:
        p.line([points[edge[0], 0], points[edge[1], 0]], 
               [points[edge[0], 1], points[edge[1], 1]], 
               line_width=2, color='black')

    show(p)

    plot_graph(points, graph)

    print(graph)

    return points, graph,feature_df



def make_primitive_Graph():
    FlatFin_folder_list = os.listdir(FlatFin_path)
    FlatFin_folder_list = [item for item in FlatFin_folder_list if os.path.isdir(os.path.join(FlatFin_path, item))]
    data_list = []
    for FlatFin_folder in FlatFin_folder_list:
        print(FlatFin_folder)
        FlatFin_dir_path = os.path.join(FlatFin_path, FlatFin_folder)
        MetaData = get_JSON(FlatFin_dir_path)
        if 'Thickness_MetaData' not in MetaData:
            continue
        MetaData = MetaData['Thickness_MetaData']

        data_list.append({
            'folder_name': FlatFin_folder,
            'file_name': MetaData['Surface file'],
            'condition': MetaData['condition'],
            'time in hpf': MetaData['time in hpf'],
            'genotype': MetaData['genotype'],
            'experimentalist': MetaData['experimentalist']
        })
    df = pd.DataFrame(data_list)

    print(df)
    
    make_path(Diffeo_path)
    points, graph,feature_df = get_primitive_Graph(df)
    print(feature_df)
    
    # Create a new DataFrame to store edges
    edge_list = []
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if node < neighbor:  # To avoid duplicates
                surface_area_1 = feature_df.iloc[node]['surface_area']
                surface_area_2 = feature_df.iloc[neighbor]['surface_area']
                
                # Determine which node has the smaller surface area
                if surface_area_1 < surface_area_2:
                    edge_list.append({
                        'node_1': feature_df.iloc[node]['folder_name'],
                        'node_2': feature_df.iloc[neighbor]['folder_name']
                    })
                else:
                    edge_list.append({
                        'node_1': feature_df.iloc[neighbor]['folder_name'],
                        'node_2': feature_df.iloc[node]['folder_name']
                    })

    edges_df = pd.DataFrame(edge_list, columns=['node_1', 'node_2'])

    print(edges_df)  # Output the new DataFrame
    edges_df.to_hdf(os.path.join(Diffeo_path,'primitive_similarity.h5'), key='data', mode='w')

def main():
    make_primitive_Graph()

if __name__ == "__main__":
    main()
