import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import pyvista as pv
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS
from bokeh.transform import factor_mark, linear_cmap
from bokeh.palettes import Viridis256
from bokeh.models import LinearColorMapper, ColorBar
import shutil

from config import *
from IO import *
from find_diffeo_hirarchical_correct import calc_normals

def get_property_df():
    FlatFin_folder_list = os.listdir(FlatFin_path)
    FlatFin_folder_list = [item for item in FlatFin_folder_list if os.path.isdir(os.path.join(FlatFin_path, item))]
    data_list = []
    for FlatFin_folder in FlatFin_folder_list:
        #print(FlatFin_folder)
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
    for i, row in df.iterrows():
        G.add_edge(row['init_folder'], row['target_folder'], weight=row['diff_energy'])

    # Retrieve property details for hover tool
    property_df = get_property_df()

    # Add node attributes to the graph from property_df
    for i, row in property_df.iterrows():
        if row['folder_name'] in G.nodes:
            G.nodes[row['folder_name']].update({
                'file_name': row['file_name'],
                'condition': row['condition'],
                'time in hpf': row['time in hpf'],
                'genotype': row['genotype'],
                'experimentalist': row['experimentalist']
            })

    # Use NetworkX for layout
    pos = nx.spring_layout(G,iterations=100)  # You can choose other layouts like 'circular_layout', 'shell_layout', etc.

    # Prepare data for Bokeh
    nodes = list(G.nodes())
    x = [pos[node][0] for node in nodes]
    y = [pos[node][1] for node in nodes]

    source = ColumnDataSource(data=dict(
        x=x,
        y=y,
        folder_name=nodes,
        file_name=[G.nodes[node].get('file_name', '') for node in nodes],
        condition=[G.nodes[node].get('condition', '') for node in nodes],
        time_in_hpf=[G.nodes[node].get('time in hpf', '') for node in nodes],
        genotype=[G.nodes[node].get('genotype', '') for node in nodes],
        experimentalist=[G.nodes[node].get('experimentalist', '') for node in nodes],
    ))

    # Create Bokeh figure
    p = figure(title="Network Graph Visualization", tools="pan,wheel_zoom,box_zoom,reset,tap")

    # Visualize nodes with scatter plot
    conditions = property_df['condition'].unique().tolist()
    markers = ['circle', 'square', 'triangle']  # Example marker list
    p.scatter('x', 'y', source=source,
              fill_alpha=0.6, size=10,
              marker=factor_mark('condition', markers, conditions),
              color=linear_cmap('time_in_hpf', Viridis256, property_df['time in hpf'].min(), property_df['time in hpf'].max()))

    # Add hover tool to show details
    hover = HoverTool()
    hover.tooltips = [
        ("Folder Name", "@folder_name"),
        ("File Name", "@file_name"),
        ("Condition", "@condition"),
        ("Time in hpf", "@time_in_hpf"),
        ("Genotype", "@genotype"),
        ("Experimentalist", "@experimentalist"),
    ]
    p.add_tools(hover)

    # Draw edges as lines between points, using edge weights for color mapping
    palette = Viridis256
    stress_min = df['diff_energy'].min()
    stress_max = df['diff_energy'].max()
    color_mapper = LinearColorMapper(palette=palette, low=stress_min, high=stress_max)

    for edge in G.edges(data=True):
        start_node, end_node = edge[0], edge[1]
        i = nodes.index(start_node)
        j = nodes.index(end_node)
        stress = edge[2]['weight']
        
        line_color = palette[int((stress - stress_min) / (stress_max - stress_min) * (len(palette) - 1))]
        
        p.line([x[i], x[j]], [y[i], y[j]], line_width=2, color=line_color)

    # Add color bar to indicate stress levels
    color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0))
    p.add_layout(color_bar, 'right')

    # Show the plot
    show(p)

def plot_debugg(mesh_init,mesh_target,ED_folder_path,show_target=True,):
    faces = mesh_init.faces.reshape((-1, 4))[:, 1:4]  # Extract triangular faces
    init_vertex_points = mesh_init.points
    x = init_vertex_points[faces]  # Shape: (N_f, 3, 3)
    N_init_1 = np.array(calc_normals(x))  # Normals on each face (N_f, 3)
    N_init_2 = np.cross(mesh_init.point_data['direction_1'],mesh_init.point_data['direction_2'])
    deformed_vertex_points = mesh_init.point_data["deformed"]
    x = deformed_vertex_points[faces]  # Shape: (N_f, 3, 3)
    N_def = np.array(calc_normals(x))
    for i, face in enumerate(faces):
        face_normals = N_init_2[face]
        face_center_normal = np.mean(face_normals, axis=0)
        if np.dot(N_init_1[i], face_center_normal) < 0:
            # Flip both N_init_1 and N_def for consistency
            N_init_1[i] = -N_init_1[i]
            N_def[i] = -N_def[i]
    arrow_scale=10
    mesh_deformed=mesh_init.copy()
    mesh_deformed.points=mesh_init.point_data["deformed"]
    p = pv.Plotter()
    p.add_mesh(mesh_deformed, color='red', show_edges=True)
    points=pv.PolyData(mesh_deformed.cell_centers().points)
    points["vectors"] = N_def * arrow_scale
    points.set_active_vectors("vectors")
    p.add_mesh(points.arrows, color='red')
    
    if show_target:
        #mesh_target.points=mesh_target.points+np.array([0,0,50])
        p.add_mesh(mesh_target, color='green', show_edges=True)



    def key_callback(key):
        print(key)
        if key == 'd':
            print(f"Deleting")
            if os.path.exists(ED_folder_path):
                try:
                    shutil.rmtree(ED_folder_path)
                except Exception as e:
                    print(f"Error: {e}")
            p.close()
            return
        if key == 'u':
            print(f"unsure")
            status_file = os.path.join(ED_folder_path, 'shown.txt')
            if os.path.exists(status_file):
                try:
                    os.remove(status_file)
                except Exception as e:
                    print(f"Error: {e}") 
            p.close()
            return


    # Add the key press callback to the plotter
    p.add_key_event('d', lambda: key_callback('d'))  # Bind 'd' to delete
    p.add_key_event('u', lambda: key_callback('u'))  # Bind 'u' to skip
    p.add_key_event('q', lambda: key_callback('q'))  # Bind 'q' to close without deleting

    try:
        p.show()
    except Exception as e:
        print(f"Error: {e}") 
    print(f"done")
    return

def plot(skip_shown=True):
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

        status_file = os.path.join(ED_folder_path, 'shown.txt')
        if skip_shown:    
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = f.read().strip()
                    if status == 'shown':
                        print(f'Skipping {ED_folder}, already shown.')
                        continue

        init_folder_dir=os.path.join(FlatFin_path,init_folder)
        init_mesh=pv.read(os.path.join(init_folder_dir,get_JSON(init_folder_dir)['Thickness_MetaData']['Surface file']))

        target_folder_dir=os.path.join(FlatFin_path,target_folder)
        target_mesh=pv.read(os.path.join(target_folder_dir,get_JSON(target_folder_dir)['Thickness_MetaData']['Surface file']))

        print(Diffeo.shape)
        print(init_mesh)
        print(target_mesh)
        init_mesh.point_data['deformed']=Diffeo

        with open(status_file, 'w') as f:
            f.write('shown')

        print(plot_debugg(init_mesh,target_mesh,ED_folder_path))
        # # init_mesh.plot()
        # # target_mesh.plot()
        # init_mesh.points=Diffeo
        # # init_mesh.plot()
        # plotter = pv.Plotter()
        # plotter.add_mesh(target_mesh, color='blue', label='Target')
        # plotter.add_mesh(init_mesh, color='red', label='Defomed')
        # plotter.add_legend()
        # plotter.show()


if __name__ == "__main__":
    #main()
    plot()