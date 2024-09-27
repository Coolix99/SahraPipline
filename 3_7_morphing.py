import pandas as pd
import networkx as nx
import pyvista as pv
from typing import List
import git
from simple_file_checksum import get_checksum
from scipy.spatial import cKDTree
import plotly.graph_objects as go

from config import *
from IO import *

def get_FlatFin_df():
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

def extract_faces(faces):
    """
    Extracts the indices of the vertices from the PyVista 1D faces array.
    Assumes triangular faces.
    """
    n_faces = faces[0::4]  # number of vertices per face, but assuming triangles here (should be 3 for each face)
    i = faces[1::4]
    j = faces[2::4]
    k = faces[3::4]
    return i, j, k

def create_morphing_movie(init_mesh, diff_arr, target_mesh, steps=30,save_path=None):
    """
    Creates a morphing animation between two meshes (init and target) using linear interpolation of diff_arr.
    Permanently displays both init_mesh and target_mesh.
    """

    # Extract the vertex indices for init_mesh and target_mesh faces
    i_init, j_init, k_init = extract_faces(init_mesh.faces)
    i_target, j_target, k_target = extract_faces(target_mesh.faces)

    # Calculate combined ranges for axes
    all_points = np.vstack([init_mesh.points, target_mesh.points])
    x_range = [all_points[:, 0].min(), all_points[:, 0].max()]
    y_range = [all_points[:, 1].min(), all_points[:, 1].max()]
    z_range = [all_points[:, 2].min(), all_points[:, 2].max()]
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
    x_mid = (x_range[0] + x_range[1]) / 2
    y_mid = (y_range[0] + y_range[1]) / 2
    z_mid = (z_range[0] + z_range[1]) / 2
    x_range = [x_mid - max_range / 2, x_mid + max_range / 2]
    y_range = [y_mid - max_range / 2, y_mid + max_range / 2]
    z_range = [z_mid - max_range / 2, z_mid + max_range / 2]
    # Set up the figure
    fig = go.Figure()

    # Plot the initial mesh permanently
    fig.add_trace(go.Mesh3d(
        x=init_mesh.points[:, 0],
        y=init_mesh.points[:, 1],
        z=init_mesh.points[:, 2],
        i=i_init,
        j=j_init,
        k=k_init,
        color='blue',
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.9, fresnel=0.1),  # Lighting effects
        lightposition=dict(x=100, y=100, z=100),  # Light position
        contour=dict(show=True, color='black', width=3),
        opacity=1.0,
        name="Initial Mesh",
        showscale=False  # Optional: hide the color scale
    ))

    # Plot the target mesh permanently (with its own face indices)
    fig.add_trace(go.Mesh3d(
        x=target_mesh.points[:, 0],
        y=target_mesh.points[:, 1],
        z=target_mesh.points[:, 2],
        i=i_target,
        j=j_target,
        k=k_target,
        color='red',
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.9, fresnel=0.1),  # Lighting effects
        lightposition=dict(x=100, y=100, z=100),  # Light position
        contour=dict(show=True, color='black', width=3),
        opacity=1.0,
        name="Target Mesh",
        showscale=False  # Optional: hide the color scale
    ))

    # Linearly interpolate between init_mesh and target_mesh using diff_arr
    frames = []
    for t in np.linspace(0, 1, steps):
        # Interpolate the mesh points
        interpolated_points = (1 - t) * init_mesh.points + t * diff_arr
        
        # Create a frame for each interpolation step
        frame = go.Frame(data=[go.Mesh3d(
            x=interpolated_points[:, 0], 
            y=interpolated_points[:, 1], 
            z=interpolated_points[:, 2],
            i=i_init,  # Faces' first vertex index (same as init_mesh)
            j=j_init,  # Faces' second vertex index (same as init_mesh)
            k=k_init,  # Faces' third vertex index (same as init_mesh)
            color='lightblue', 
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.9, fresnel=0.1),  # Lighting effects
            lightposition=dict(x=100, y=100, z=100),  # Light position
            contour=dict(show=True, color='black', width=3),
            opacity=1.0,
            name=f"Frame {t}",
            showscale=False
        )])
        
        frames.append(frame)

    # Set the frames for animation
    fig.frames = frames

    # Set up the layout, ensuring equal axis scaling and dynamic axis ranges
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=x_range),  # Hide X axis
            yaxis=dict(visible=False, range=y_range),  # Hide Y axis
            zaxis=dict(visible=False, range=z_range),  # Hide Z axis
            bgcolor="black",
            aspectmode="cube"  # Ensures equal axis spacing
        ),
        updatemenus=[dict(type="buttons", showactive=False,
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, dict(frame=dict(duration=50, redraw=True), 
                                                         fromcurrent=True)])])]
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Animation saved as {save_path}")
    fig.show()
    
def main():
    diffeos_df = pd.read_hdf(os.path.join(ElementaryDiffeos_path,'elementary.h5'))
    FlatFin_df=get_FlatFin_df()

    merged_df = diffeos_df.merge(FlatFin_df, left_on='init_folder', right_on='folder_name', suffixes=('_init', '_target'))
    merged_df = merged_df.merge(FlatFin_df, left_on='target_folder', right_on='folder_name', suffixes=('_init', '_target'))
    filtered_df = merged_df[(merged_df['time in hpf_init'] == 132) & (merged_df['time in hpf_target'] == 132)]

    n=3
    smallest_energy_entry = filtered_df.nsmallest(n, 'diff_energy')

    folder_path=os.path.join(FlatFin_path,smallest_energy_entry['init_folder'].iloc[n-1])
    surface_file_name=get_JSON(folder_path)['Thickness_MetaData']['Surface file']
    init_mesh=pv.read(os.path.join(folder_path,surface_file_name))

    folder_path=os.path.join(FlatFin_path,smallest_energy_entry['target_folder'].iloc[n-1])
    surface_file_name=get_JSON(folder_path)['Thickness_MetaData']['Surface file']
    target_mesh=pv.read(os.path.join(folder_path,surface_file_name))

    diff_folder_path=os.path.join(ElementaryDiffeos_path,smallest_energy_entry['diff_folder'].iloc[n-1])
    diff_arr=np.load(os.path.join(diff_folder_path,get_JSON(diff_folder_path)['MetaData_Diffeo']['Diffeo file']))
    
    create_morphing_movie(init_mesh, diff_arr, target_mesh, steps=60)
    # save_path=r'C:\Users\kotzm\Documents\morphing.html'
if __name__ == "__main__":
    main()