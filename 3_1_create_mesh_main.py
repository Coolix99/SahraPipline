import numpy as np
from typing import List
import napari
import os
import git
from simple_file_checksum import get_checksum
import re
from scipy.ndimage import label
from skimage import measure
import pyvista as pv

from config import *
from IO import *

def extract_condition(s):
    # Using regular expressions to find 'dev' or 'reg'
    match = re.search(r'(dev|reg)', s)
    if match:
        return match.group(1)
    else:
        # If neither 'dev' nor 'reg' is found, raise an error
        raise ValueError(f"No 'dev' or 'reg' found in string: {s}")

def extract_hpf(s):
    """
    Extracts an integer immediately preceding 'hpf' in a given string.

    Args:
    s (str): The input string from which to extract the number.

    Returns:
    int: The extracted integer, or None if no such pattern exists.
    """
    match = re.search(r'(\d+)hpf', s)
    return int(match.group(1)) if match else None

def find_correct_orientation(signal, mask):
    """
    Flips the mask along different axes to ensure all non-zero pixels of the signal image are inside the mask.

    Args:
    signal (np.array): The signal image array.
    mask (np.array): The mask image array.

    Returns:
    np.array: The correctly oriented mask or None if no orientation meets the condition.
    """
    # Check each possible flip configuration

    ind_sig=np.where((signal > 0))
    for axis in [None, 0, 1, (0, 1)]:
        flipped_mask = np.flip(mask, axis=axis) if axis is not None else mask.copy()
        print(axis)
        # Check if all non-zero pixels of the signal are within the mask
        if (np.sum(flipped_mask[ind_sig]>1)/ind_sig[0].shape[0])>0.9:
            return flipped_mask  # Return the correctly oriented mask

    return None  # Return None if no valid configuration is found

def extract_largest_component(mask):
    """
    Extracts the largest connected component from a binary mask.

    Args:
    mask (np.array): The binary mask array.

    Returns:
    np.array: A binary mask of the largest connected component.
    """
    # Label all connected components
    labeled_mask, num_features = label(mask)
    
    # Find the sizes of all components
    sizes = np.bincount(labeled_mask.ravel())
    
    # Determine the label of the largest component (excluding the background, which is label 0)
    largest_label = sizes[1:].argmax() + 1
    
    # Create a mask for the largest connected component
    largest_component = (labeled_mask == largest_label)
    
    return largest_component

def create_mesh_from_mask(mask):
    """
    Applies the Marching Cubes algorithm to a mask and converts it into a PyVista mesh.
    Ensures no holes by padding the mask before processing.

    Args:
    mask (np.array): The binary mask array (3D).

    Returns:
    pv.PolyData: A PyVista mesh of the triangulated surface.
    """
    # Pad the mask to prevent holes at the edges
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)

    # Perform the Marching Cubes algorithm on the padded mask
    verts, faces, _, _ = measure.marching_cubes(padded_mask, level=0)

    # Adjust the vertices by subtracting one to compensate for the padding
    verts -= 1

    # Verts are the coordinates for each vertex
    # Faces are the indices of vertices that form each triangle
    # Convert faces to the correct format
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])

    # Create a PyVista mesh (polydata object)
    mesh = pv.PolyData(verts, faces)

    return mesh

def reduce_and_smooth_mesh(orig_mesh:pv.PolyData, desired_density_per_unit_area = 0.03, num_iter=200):
    """
    Reduces the number of vertices in the mesh and applies a smoothing algorithm.

    Args:
    mesh (pv.PolyData): The input PyVista mesh.
    reduction_factor (float): The fraction to reduce the mesh by (0.5 means reduce by 50%).
    num_iter (int): The number of iterations to use in the smoothing process.

    Returns:
    pv.PolyData: The reduced and smoothed mesh.
    """
    # Reduce the number of faces/vertices
    mesh = orig_mesh.decimate_pro(0.5)

    # Smooth the mesh
    mesh = mesh.smooth(n_iter=20, relaxation_factor=0.2)

    while True:
        current_vertex_count = mesh.n_points
        total_surface_area = mesh.area
        target_vertex_count = desired_density_per_unit_area * total_surface_area
        print(current_vertex_count,target_vertex_count)
        if target_vertex_count > current_vertex_count:
            break
        decimation_factor = target_vertex_count / current_vertex_count - 0.1
        if decimation_factor<0.1:
            decimation_factor = 0.1 
        mesh = mesh.decimate(1-decimation_factor)

    print('finished decimate')
    mesh=mesh.smooth(n_iter=num_iter, relaxation_factor=0.5)

    return mesh

def plot_and_capture_view(mesh):
    """
    Plots the mesh interactively and captures the camera settings after user interaction.

    Args:
    mesh (pv.PolyData): The PyVista mesh to plot.

    Returns:
    tuple: A tuple containing camera settings (position, focal point, view up vector).
    """
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white')
    plotter.show(auto_close=False)  # Keep the plotter window open

    # After user interaction, when they close the window
    cam_pos = plotter.camera_position
    plotter.close()

    return cam_pos

def remove_cells_by_angle(mesh, view_direction, angle_threshold=120):
    """
    Removes cells from a mesh where the angle with the view direction is greater than a given threshold.

    Args:
    mesh (pv.PolyData): The mesh with computed normals.
    view_direction (np.array): The viewing direction vector, should be normalized.
    angle_threshold (float): The threshold angle in degrees.

    Returns:
    pv.PolyData: A new mesh with only the cells with angles <= threshold angle to the view direction.
    """
    # Ensure the normals and view direction are normalized
    view_direction /= np.linalg.norm(view_direction)
    cell_normals = mesh.cell_normals / np.linalg.norm(mesh.cell_normals, axis=1, keepdims=True)

    # Calculate the dot product between each normal and the view direction
    dot_products = np.einsum('ij,j->i', cell_normals, view_direction)

    # Calculate the angle in degrees between the normals and the view direction
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0)) * (180 / np.pi)

    # Filter cells where the angle is less than or equal to the threshold
    good_cells = angles <= angle_threshold

    # Extract cells that satisfy the condition
    good_cell_indices = mesh.faces.reshape(-1, 4)[good_cells]

    # Create a new mesh with only the valid cells
    new_mesh = pv.PolyData(mesh.points, np.hstack([np.full((len(good_cell_indices), 1), 3), good_cell_indices[:, 1:]]))

    cleaned_mesh = new_mesh.clean()

    # Fill holes in the mesh to close any gaps
    filled_mesh = cleaned_mesh.fill_holes(0.2)

    # Smooth the mesh to improve the appearance
    smooth_mesh = filled_mesh.smooth(n_iter=10, relaxation_factor=0.1)

    return smooth_mesh

def evalStatus_mesh(res_path):
    MetaData=get_JSON(res_path)
    if not 'Mesh_MetaData' in MetaData:
        return MetaData

    if not MetaData['Mesh_MetaData']['Mesh version']==Mesh_version:
        return MetaData  
    
    return False

def create_mesh():
    masks_dir_path=os.path.join(EpFlat_path,'peeled_mask')
    signal_dir_path=os.path.join(EpFlat_path,'signal_upperlayer')
    mask_list=os.listdir(masks_dir_path)
    for mask in mask_list:
        data_name=mask[:-11]
        print(data_name)
        signal_path=os.path.join(signal_dir_path,data_name+'.tif')
        if not os.path.exists(signal_path):
            print('no sig')
            continue
        mask_path=os.path.join(masks_dir_path,mask)

        res_path=os.path.join(EpSeg_path,data_name)
        PastMetaData=evalStatus_mesh(res_path)
        if not isinstance(PastMetaData,dict):
            continue
        make_path(res_path)

        mask_img,scale=getImage_Meta(mask_path)
        sig_img=getImage(signal_path)

        condition=extract_condition(data_name)
        time=extract_hpf(data_name)

        flipped_mask=find_correct_orientation(sig_img, mask_img)
        flipped_mask=extract_largest_component(flipped_mask)

        mesh=create_mesh_from_mask(flipped_mask)
        mesh=reduce_and_smooth_mesh(mesh)
        print('reduction finished,start compute normals')
        mesh.compute_normals(point_normals=False, cell_normals=True, auto_orient_normals=True, flip_normals=False)
        print('normals finished')
        view=plot_and_capture_view(mesh)
        view_direction=view[2]
        mesh=remove_cells_by_angle(mesh, view_direction)
        #mesh.plot(show_edges=True)
        mesh.points=mesh.points*scale
        mesh=mesh.subdivide_adaptive(max_edge_len=3.0,inplace=True)
        mesh.compute_normals(point_normals=False, cell_normals=True, auto_orient_normals=True, flip_normals=False)
        mesh.plot(show_edges=True)

        mesh_file_name=data_name+'_mesh.vtk'
        mesh_file=os.path.join(res_path,mesh_file_name)
        mesh.save(mesh_file)
            
        MetaData_Mesh={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_Mesh['git hash']=sha
        MetaData_Mesh['git repo']='Sahrapipline'
        MetaData_Mesh['Mesh version']=Mesh_version
        MetaData_Mesh['Mesh file']=mesh_file_name
        MetaData_Mesh['scales']=[scale[0],scale[1],scale[2]]
        MetaData_Mesh['condition']=condition
        MetaData_Mesh['time in hpf']=time
        MetaData_Mesh['experimentalist']='Sahra'
        MetaData_Mesh['genotype']='WT'
        check_mesh=get_checksum(mesh_file, algorithm="SHA1")
        MetaData_Mesh['output mesh checksum']=check_mesh
        writeJSON(res_path,'Mesh_MetaData',MetaData_Mesh)       

if __name__ == "__main__":
    create_mesh()
