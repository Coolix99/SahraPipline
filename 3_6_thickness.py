import pandas as pd
import pyvista as pv 
import numpy as np
import git
from simple_file_checksum import get_checksum

from config import *
from IO import *

def filter_nested_matches(matches):
    """
    Filters out matches where one match is a substring of another.
    Retains only the longer string if one is contained in another.

    Args:
    - matches (list of str): The list of matches.

    Returns:
    - list of str: Filtered list of matches.
    """
    filtered_matches = []
    for match in matches:
        # Check if match is not a substring of any other longer match
        if not any(match in other and match != other for other in matches):
            filtered_matches.append(match)
    return filtered_matches


def find_match(EDcells_folder, ED_cell_props_folder_list):
    matches = []
    for membrane in ED_cell_props_folder_list:
        #print(membrane)
        cutoff_membrane=membrane
        if "_2024_" in cutoff_membrane:
            cutoff_membrane = cutoff_membrane.split("_2024_")[0]
        if "_Stitch" in cutoff_membrane:
            cutoff_membrane = cutoff_membrane.split("_Stitch")[0]

        if cutoff_membrane in EDcells_folder:
            matches.append(membrane)  # Append the full string if matched

    return filter_nested_matches(matches)

def get_thickness_cellLayers(mesh:pv.PolyData,EDcells_img,scales_zyx):
    n = mesh.point_normals
    p = mesh.points

    d_max = 40
    dd = 0.5

    d = np.linspace(-d_max, d_max, int(2 * d_max / dd) + 1)

    n_expanded = n[:, np.newaxis, :]  # Shape: Nx1x3
    d_expanded = d[np.newaxis, :, np.newaxis]  # Shape: 1xMx1

    # Compute positions p + n * d
    pos = p[:, np.newaxis, :] + n_expanded * d_expanded  # NxMx3

    # Convert to pixel positions in EDcells_img
    pixle_pos = (pos / scales_zyx + 0.5).astype(np.uint16)  # NxMx3

    # Ensure positions are within image bounds
    max_bounds = np.array(EDcells_img.shape) - 1
    pixle_pos = np.clip(pixle_pos, 0, max_bounds)

    # Extract values from EDcells_img
    values = np.zeros((pixle_pos.shape[0], pixle_pos.shape[1]), dtype=EDcells_img.dtype)
    for i in range(pixle_pos.shape[0]):
        for j in range(pixle_pos.shape[1]):
            z, y, x = pixle_pos[i, j]
            values[i, j] = EDcells_img[z, y, x]

    print(values.shape)

    binary_values = (values > 0).astype(np.float32)  # Convert to binary (0 or 1)
    centers = np.zeros((values.shape[0], 3), dtype=np.float32)  # Nx3 result array
    thickness = np.zeros((values.shape[0]), dtype=np.float32)
    # Compute the real-space center for each N
    for i in range(values.shape[0]):
        if binary_values[i].sum() > 4:  # Avoid division by zero
            # Weighted center along the M axis
            center_m = np.average(d, weights=binary_values[i])
            centers[i] = mesh.points[i] + center_m * mesh.point_normals[i]
            
            non_zero_indices = np.where(binary_values[i] > 0)[0]
            min_d = d[non_zero_indices.min()]  # Smallest d corresponding to non-zero value
            max_d = d[non_zero_indices.max()]  # Largest d corresponding to non-zero value
            thickness[i] = max_d - min_d

    # Second Task: Count unique objects for each N
    object_counts = np.array([len(np.unique(row[row > 0])) for row in values])  # Count unique values (non-zero)
   
    print(object_counts.shape)

    # v=mesh.points
    # f=mesh.faces.reshape((-1,4))[:,1:4]

    # import napari
    # viewer=napari.Viewer()

    # viewer.add_surface((v,f,object_counts),name='mesh with count')
    # viewer.add_surface((v,f,thickness),name='thickness')
    # viewer.add_labels(EDcells_img,name='ED',scale=scales_zyx)
    # viewer.add_points(centers,name='centers')

    # napari.run()

    return centers,object_counts,thickness

def main():
    EDcells_folder_list= [item for item in os.listdir(ED_cells_path) if os.path.isdir(os.path.join(ED_cells_path, item))]
    ED_cell_props_folder_list= [item for item in os.listdir(ED_cell_props_path) if os.path.isdir(os.path.join(ED_cell_props_path, item))]

    for EDcells_folder in EDcells_folder_list:
        EDcells_folder_path=os.path.join(ED_cells_path,EDcells_folder)
        
        EDcellsMetaData=get_JSON(EDcells_folder_path)
        if EDcellsMetaData=={}:
            print('no EDcells')
            continue
        
        matches=find_match(EDcells_folder,ED_cell_props_folder_list)

        if len(matches)!=1:
            print(f"not one match found", len(matches))
            continue
        
        data_name=matches[0]
        print(data_name)
        ED_cell_props_folder_path=os.path.join(ED_cell_props_path,data_name)
        MetaData_EDcell_props=get_JSON(ED_cell_props_folder_path)
        if not 'MetaData_EDcell_props' in MetaData_EDcell_props:
            print('no props metadata')
            continue
        # if not 'MetaData_EDcell_top' in MetaData_EDcell_props:
        #     print('no top metadata')
        #     continue

        folder_path=os.path.join(FlatFin_path,data_name+'_FlatFin')
        FF_MetaData=get_JSON(folder_path)
        if FF_MetaData=={}:
            print('no FF_MetaData')
            continue
        surface_file_name=FF_MetaData['Thickness_MetaData']['Surface file']
        mesh=pv.read(os.path.join(folder_path,surface_file_name))
        EDcells_img=getImage(os.path.join(EDcells_folder_path,EDcellsMetaData['MetaData_EDcells']['EDcells file'])).astype(np.uint16)
        
        scales_zyx=FF_MetaData['Thickness_MetaData']['scales ZYX']

        centers,object_counts,thickness=get_thickness_cellLayers(mesh,EDcells_img,scales_zyx)
        
        saveArr(centers,os.path.join(ED_cell_props_folder_path,'centers_ED'))
        saveArr(object_counts,os.path.join(ED_cell_props_folder_path,'object_counts_ED'))
        saveArr(thickness,os.path.join(ED_cell_props_folder_path,'thickness_ED'))

        MetaData_EDthick_top={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_EDthick_top['git hash']=sha
        MetaData_EDthick_top['git repo']='Sahrapipline'
        MetaData_EDthick_top['centers file']='centers_ED'
        MetaData_EDthick_top['object_counts file']='object_counts_ED'
        MetaData_EDthick_top['thickness file']='thickness_ED'
        MetaData_EDthick_top['condition']=FF_MetaData['Thickness_MetaData']['condition']
        MetaData_EDthick_top['time in hpf']=FF_MetaData['Thickness_MetaData']['time in hpf']
        MetaData_EDthick_top['genotype']=FF_MetaData['Thickness_MetaData']['genotype']
        MetaData_EDthick_top['experimentalist']=FF_MetaData['Thickness_MetaData']['experimentalist']
        MetaData_EDthick_top['scales ZYX']=FF_MetaData['Thickness_MetaData']['scales ZYX']
        check=get_checksum(os.path.join(ED_cell_props_folder_path,'centers_ED.npy'), algorithm="SHA1")+\
            get_checksum(os.path.join(ED_cell_props_folder_path,'centers_ED.npy'), algorithm="SHA1")+\
            get_checksum(os.path.join(ED_cell_props_folder_path,'centers_ED.npy'), algorithm="SHA1")
        MetaData_EDthick_top['EDthik checksum']=check
        writeJSON(ED_cell_props_folder_path,'MetaData_EDthik',MetaData_EDthick_top)      
        
        

if __name__ == "__main__":
    main()