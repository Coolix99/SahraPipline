from functools import reduce
import pandas as pd
import git
import napari
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

def combine_metadata(meta_props, meta_membrane):
    # Define the important keys
    important_keys = ['scales ZYX', 'condition', 'time in hpf', 'experimentalist', 'genotype']
    
    combined_metadata = {}
    
    for key in important_keys:
        value_props = meta_props.get(key)
        value_membrane = meta_membrane.get(key)
        
        if value_props is not None and value_membrane is not None:
            if value_props != value_membrane:
                raise ValueError(f"Conflict for key '{key}': {value_props} != {value_membrane}")
            combined_metadata[key] = value_props  # Both values are equal
        elif value_props is not None:
            combined_metadata[key] = value_props
        elif value_membrane is not None:
            combined_metadata[key] = value_membrane
        else:
            raise ValueError(f"Key '{key}' is missing in both metadata objects")
    
    return combined_metadata

def plot_propertie(EDcells_img,df_prop,df_top):
    # Start Napari viewer
    viewer = napari.Viewer()

    # Add the segmented image
    #viewer.add_labels(EDcells_img, name="Segmented Objects")

    # Add solidity
    # df_prop = df_prop[df_prop['Solidity'].notnull()]  
    # df_prop = df_prop[np.isfinite(df_prop['Solidity'])] 
    # max_label = EDcells_img.max()  # Find the maximum label value in the image
    # solidity_map = np.zeros(max_label + 1, dtype=float)  # +1 to handle all labels
    # solidity_map[df_prop['Label']] = df_prop['Solidity']
    # solidity_colored_img = solidity_map[EDcells_img]

    # solidity_colored_img_normalized = (solidity_colored_img - solidity_colored_img.min()) / (
    #     solidity_colored_img.max() - solidity_colored_img.min()
    # )
    # # Add solidity-based color layer
    # viewer.add_image(
    #     solidity_colored_img_normalized,
    #     colormap="gist_earth",
    #     name="Solidity Coloring",
    #     blending="additive",
    # )

    # centroids = df_prop[['centroids Z','centroids Y', 'centroids X']].to_numpy()
    # viewer.add_points(
    #     centroids,
    #     size=5,
    #     face_color='cyan',
    #     name='Centroids',
    # )

    # # Add line connections from df_top
    # lines = []
    # for i in range(len(df_top)):
    #     label1 = df_top['Label 1'].iloc[i]
    #     label2 = df_top['Label 2'].iloc[i]
    #     # Get the coordinates of the centroids
    #     point1 = df_prop[df_prop['Label'] == label1][['centroids Z', 'centroids Y', 'centroids X']].iloc[0].to_numpy()
    #     point2 = df_prop[df_prop['Label'] == label2][['centroids Z', 'centroids Y', 'centroids X']].iloc[0].to_numpy()
    #     lines.append([point1, point2])

    # viewer.add_shapes(
    #     lines,
    #     shape_type='line',
    #     edge_color='yellow',
    #     name='Connections',
    # )


    centroids = df_prop[['centroids_scaled Z','centroids_scaled Y', 'centroids_scaled X']].to_numpy()
    viewer.add_points(
        centroids,
        size=5,
        face_color='cyan',
        name='Centroids',
    )

    # Add line connections from df_top
    lines = []
    for i in range(len(df_top)):
        label1 = df_top['Label 1'].iloc[i]
        label2 = df_top['Label 2'].iloc[i]
        # Get the coordinates of the centroids
        point1 = df_prop[df_prop['Label'] == label1][['centroids_scaled Z', 'centroids_scaled Y', 'centroids_scaled X']].iloc[0].to_numpy()
        point2 = df_prop[df_prop['Label'] == label2][['centroids_scaled Z', 'centroids_scaled Y', 'centroids_scaled X']].iloc[0].to_numpy()
        lines.append([point1, point2])

    viewer.add_shapes(
        lines,
        shape_type='line',
        edge_color='yellow',
        name='Connections',
    )

    # Start the Napari application
    napari.run()

def main():
    EDcells_folder_list= [item for item in os.listdir(ED_cells_path) if os.path.isdir(os.path.join(ED_cells_path, item))]
    ED_cell_props_folder_list= [item for item in os.listdir(ED_cell_props_path) if os.path.isdir(os.path.join(ED_cell_props_path, item))]

    for EDcells_folder in EDcells_folder_list:
        print(EDcells_folder)
        if not 'reg' in EDcells_folder:
            continue
        if not '144h' in EDcells_folder:
            continue
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
        if not 'MetaData_EDcell_top' in MetaData_EDcell_props:
            print('no top metadata')
            continue

                

        EDcells_img=getImage(os.path.join(EDcells_folder_path,EDcellsMetaData['MetaData_EDcells']['EDcells file'])).astype(np.uint16)
        df_prop = pd.read_hdf(os.path.join(ED_cell_props_folder_path,MetaData_EDcell_props['MetaData_EDcell_props']['EDcells file']), key='data') #might change
        df_top = pd.read_hdf(os.path.join(ED_cell_props_folder_path,MetaData_EDcell_props['MetaData_EDcell_top']['EDcell_top file']), key='data')

        print(df_prop)
        print(df_top)
        plot_propertie(EDcells_img,df_prop,df_top)
        
    

if __name__ == "__main__":
    main()