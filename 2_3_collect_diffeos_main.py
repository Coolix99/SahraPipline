import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS
from bokeh.transform import factor_mark, linear_cmap
from bokeh.palettes import Viridis256
from bokeh.models import LinearColorMapper, ColorBar


from config import *
from IO import *

def get_property_df():
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
    return df

def plot_stress_histogram(stresses):
    
    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(stresses, bins=20, color='blue', edgecolor='black')
    plt.title("Histogram of Edge Stress Values")
    plt.xlabel("Stress (Relative Length Change)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def main():
    # Load your elementary diffeomorphisms data
    ED_folder_list = [folder for folder in os.listdir(ElementaryDiffeos_path) if os.path.isdir(os.path.join(ElementaryDiffeos_path, folder))]
    data_list = []
    for ED_folder in ED_folder_list:
        ED_folder_path = os.path.join(ElementaryDiffeos_path, ED_folder)
        MetaData = get_JSON(ED_folder_path)
        if not 'MetaData_Diffeo' in MetaData:
            continue
        data_list.append({
            'diff_folder': ED_folder,
            'init_folder': MetaData['MetaData_Diffeo']['init_folder'],
            'target_folder': MetaData['MetaData_Diffeo']['target_folder'],
            'diff_energy': MetaData['MetaData_Diffeo']['diff_energy']
        })
    
    df = pd.DataFrame(data_list)
    df.to_hdf(os.path.join(ElementaryDiffeos_path, 'elementary.h5'), key='data', mode='w')
    
    # Create a graph from the dataframe
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['init_folder'], row['target_folder'], weight=row['diff_energy'])
    
    # Compute the shortest path distance matrix
    def create_sparse_distance_matrix(G):
        # Initialize a distance matrix with infinity
        n = len(G)
        dist_matrix = np.full((n, n), 0)
        
        # Map node labels to indices
        node_index = {node: idx for idx, node in enumerate(G.nodes())}
        
        # Fill the matrix with the edge weights where edges exist
        for u, v, data in G.edges(data=True):
            i, j = node_index[u], node_index[v]
            dist_matrix[i, j] = data['weight']
            dist_matrix[j, i] = data['weight']  # Assuming the graph is undirected
        return dist_matrix
    
    dist_matrix = create_sparse_distance_matrix(G)
    print(dist_matrix)
    
    from sklearn.manifold import Isomap,LocallyLinearEmbedding,SpectralEmbedding
    #Perform Isomap embedding for 2 dimensions
    isomap = MDS(n_components=2, dissimilarity='precomputed', random_state=42,verbose=10,max_iter=1000,metric=False,eps=1e-10)
    embedding_2d = isomap.fit_transform(dist_matrix)
    
    #Retrieve property details for hover tool
    property_df = get_property_df()
    
    #Match the embeddings with the folder names in property_df
    embedding_df = pd.DataFrame(embedding_2d, columns=['x', 'y'])
    embedding_df['folder_name'] = list(G.nodes)
    merged_df = pd.merge(embedding_df, property_df, on='folder_name')
    
    #Prepare the data for Bokeh
    source = ColumnDataSource(merged_df)
    conditions = merged_df['condition'].unique()
    markers = ['circle', 'triangle', 'square', 'diamond', 'inverted_triangle']
    
    #Compute original and embedded distances
    original_distances = dict(nx.get_edge_attributes(G, 'weight'))
    embedded_distances = {}
    
    for edge in G.edges():
       i, j = list(G.nodes()).index(edge[0]), list(G.nodes()).index(edge[1])
       embedded_distances[edge] = np.linalg.norm(embedding_2d[i] - embedding_2d[j])
    
    #Calculate relative length change (stress)
    stresses = {}
    for edge, original_length in original_distances.items():
       embedded_length = embedded_distances[edge]
       stresses[edge] = abs(embedded_length - original_length) / original_length
    
    #Normalize stress values for color mapping
    stress_values = np.array(list(stresses.values()))
    stress_min, stress_max = stress_values.min(), stress_values.max()
    plot_stress_histogram(stress_values)
    # Create Bokeh figure
    p = figure(title="2D MDS Embedding of the Graph", tools="pan,wheel_zoom,box_zoom,reset,tap")
    
    p.scatter('x', 'y', source=source,
              fill_alpha=0.6, size=10,
              marker=factor_mark('condition', markers, conditions),
              color=linear_cmap('time in hpf', Viridis256, merged_df['time in hpf'].min(), merged_df['time in hpf'].max()))

    # Add hover tool to show details
    hover = HoverTool()
    hover.tooltips = [
        ("Folder Name", "@folder_name"),
        ("File Name", "@file_name"),
        ("Condition", "@condition"),
        ("Time in hpf", "@{time in hpf}"),
        ("Genotype", "@genotype"),
        ("Experimentalist", "@experimentalist"),
    ]
    p.add_tools(hover)
    
    # Draw edges as lines between points, color-coded by stress
    palette = Viridis256
    color_mapper = LinearColorMapper(palette=palette, low=stress_min, high=stress_max)
    
    for edge, stress in stresses.items():
        i, j = list(G.nodes()).index(edge[0]), list(G.nodes()).index(edge[1])
        # Convert stress to a color in the palette
        stress_normalized = (stress - stress_min) / (stress_max - stress_min)
        color_index = int(stress_normalized * (len(palette) - 1))
        line_color = palette[color_index]
        p.line([embedding_2d[i, 0], embedding_2d[j, 0]], 
               [embedding_2d[i, 1], embedding_2d[j, 1]], 
               line_width=2, color=line_color)
    
    # Add color bar to indicate stress levels
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0))
    p.add_layout(color_bar, 'right')

    show(p)
    
    # Analyze embedding energy for different dimensions
    dimensions = list(range(2, 10))  # Embedding into 2 to 5 dimensions
    stress_values = []
    for dim in dimensions:

        stresses = {}
        isomap = MDS(n_components=dim, dissimilarity='precomputed', random_state=42,metric=False)
        embedding = isomap.fit_transform(dist_matrix)
        print(embedding.shape)
        for edge in G.edges():
            i, j = list(G.nodes()).index(edge[0]), list(G.nodes()).index(edge[1])
            embedded_distances[edge] = np.linalg.norm(embedding[i] - embedding[j])
        for edge, original_length in original_distances.items():
            embedded_length = embedded_distances[edge]
            stresses[edge] = abs(embedded_length - original_length) / original_length
        hist_stress = np.array(list(stresses.values()))
        print(dim)
        stress_values.append(isomap.stress_)
        plot_stress_histogram(hist_stress)
    
    # Plot embedding energy (stress) as a function of dimension
    plt.figure(figsize=(8, 6))
    plt.plot(dimensions, stress_values, marker='o')
    plt.title("MDS Embedding Stress vs. Dimension")
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Stress (Energy)")
    plt.grid(True)
    plt.show()


def plot():
    ED_folder_list=[folder for folder in os.listdir(ElementaryDiffeos_path) if os.path.isdir(os.path.join(ElementaryDiffeos_path, folder))]
    #ED_folder_list=['diffeo_0a0502224f']
    for ED_folder in ED_folder_list:
        print(ED_folder)
        ED_folder_path=os.path.join(ElementaryDiffeos_path,ED_folder)
        MetaData=get_JSON(ED_folder_path)
        if not 'MetaData_Diffeo' in MetaData:
            continue
        
        Diffeo=np.load(os.path.join(ED_folder_path,MetaData['MetaData_Diffeo']["Diffeo file"]))

        init_folder=MetaData['MetaData_Diffeo']['init_folder']
        target_folder=MetaData['MetaData_Diffeo']['target_folder']

        init_folder_dir=os.path.join(FlatFin_path,init_folder)
        init_mesh=pv.read(os.path.join(init_folder_dir,get_JSON(init_folder_dir)['Thickness_MetaData']['Surface file']))

        target_folder_dir=os.path.join(FlatFin_path,target_folder)
        target_mesh=pv.read(os.path.join(target_folder_dir,get_JSON(target_folder_dir)['Thickness_MetaData']['Surface file']))

        print(Diffeo.shape)
        print(init_mesh)
        print(target_mesh)
        # init_mesh.plot()
        # target_mesh.plot()
        init_mesh.points=Diffeo
        # init_mesh.plot()
        plotter = pv.Plotter()
        plotter.add_mesh(target_mesh, color='blue', label='Target')
        plotter.add_mesh(init_mesh, color='red', label='Defomed')
        plotter.add_legend()
        plotter.show()


if __name__ == "__main__":
    main()
    #plot()