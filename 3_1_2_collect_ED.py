from functools import reduce
import pyvista as pv
import pandas as pd

import reg_prop_3d as prop3
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

def find_match(EDcells_folder, finmask_folder_list):
    matches = []
    for membrane in finmask_folder_list:
        #print(membrane)
        cutoff_membrane=membrane
        if "_2024_" in cutoff_membrane:
            cutoff_membrane = cutoff_membrane.split("_2024_")[0]
        if "_Stitch" in cutoff_membrane:
            cutoff_membrane = cutoff_membrane.split("_Stitch")[0]

        if cutoff_membrane in EDcells_folder:
            matches.append(membrane)  # Append the full string if matched

    return filter_nested_matches(matches)

def combine_metadata(meta_finmask, meta_membrane):
    # Define the important keys
    important_keys = ['scales ZYX', 'condition', 'time in hpf', 'experimentalist', 'genotype']
    
    combined_metadata = {}
    
    for key in important_keys:
        value_finmask = meta_finmask.get(key)
        value_membrane = meta_membrane.get(key)
        
        if value_finmask is not None and value_membrane is not None:
            if value_finmask != value_membrane:
                raise ValueError(f"Conflict for key '{key}': {value_finmask} != {value_membrane}")
            combined_metadata[key] = value_finmask  # Both values are equal
        elif value_finmask is not None:
            combined_metadata[key] = value_finmask
        elif value_membrane is not None:
            combined_metadata[key] = value_membrane
        else:
            raise ValueError(f"Key '{key}' is missing in both metadata objects")
    
    return combined_metadata

def getCellProps(seg,Vox_size):
    # if Image_MetaData['axes']=='ZYX':
    #     Vox_size=np.array((Image_MetaData['XYZ size in mum'][2],Image_MetaData['XYZ size in mum'][1],Image_MetaData['XYZ size in mum'][0]))
    # else:
    #     print('other ax order')

    props = prop3.regionprops_3D(seg,Vox_size)
    
    volumes=[]
    surface_areas=[]
    inertia_tensor_eigvals=[]
    soliditys=[]
    centroids=[]
    centroids_scaled=[]
    labels=[]
    for i,r in enumerate(props):
        if i%10==0:
            print(i, ' of ', len(props))
        volumes.append(r['volume'])
        surface_areas.append(r['surface_area'])
        inertia_tensor_eigvals.append(r['inertia_tensor_eigvals_scaled'])
        soliditys.append(r['solidity'])
        centroids.append(r['centroid'])
        centroids_scaled.append(np.array((centroids[-1][0]*Vox_size[0],centroids[-1][1]*Vox_size[1],centroids[-1][2]*Vox_size[2])))
        labels.append(r['label'])

    data = {'Volume': volumes, 
            'Surface Area': surface_areas,
            'Solidity': soliditys,
            'Label': labels}
    
    df = pd.DataFrame(data)
    vec_df = pd.DataFrame(inertia_tensor_eigvals, columns=['inertia_tensor_eigvals 1', 'inertia_tensor_eigvals 2', 'inertia_tensor_eigvals 3'])
    df = pd.concat([df, vec_df], axis=1)
    vec_df = pd.DataFrame(centroids, columns=['centroids Z', 'centroids Y', 'centroids X'])
    df = pd.concat([df, vec_df], axis=1)
    vec_df = pd.DataFrame(centroids_scaled, columns=['centroids_scaled Z', 'centroids_scaled Y', 'centroids_scaled X'])
    df = pd.concat([df, vec_df], axis=1)

    df = df.apply(pd.to_numeric, errors='coerce')
    print(df)
    return df

def main():
    EDcells_folder_list= [item for item in os.listdir(ED_cells_path) if os.path.isdir(os.path.join(ED_cells_path, item))]
    finmask_folder_list= [item for item in os.listdir(finmasks_path) if os.path.isdir(os.path.join(finmasks_path, item))]
    data_list = []
    for EDcells_folder in EDcells_folder_list:
        print(EDcells_folder)
        EDcells_folder_path=os.path.join(ED_cells_path,EDcells_folder)
        
        EDcellsMetaData=get_JSON(EDcells_folder_path)
        if EDcellsMetaData=={}:
            print('no EDcells')
            continue
        
        matches=find_match(EDcells_folder,finmask_folder_list)

        if len(matches)!=1:
            print(f"not one match found", len(matches))
            continue
        
        data_name=matches[0]
        print(data_name)
        finmasks_folder_path=os.path.join(finmasks_path,data_name)
        Meta_Data_finmask=get_JSON(finmasks_folder_path)
        if 'MetaData_finmasks' in Meta_Data_finmask:
            Meta_Data_finmask=Meta_Data_finmask['MetaData_finmasks']
        
        membrane_folder_path=os.path.join(membranes_path,data_name)
        Meta_Data_membrane=get_JSON(membrane_folder_path)
        if 'MetaData_membrane' in Meta_Data_membrane:
            Meta_Data_membrane=Meta_Data_membrane['MetaData_membrane']
        
        MetaData=combine_metadata(Meta_Data_finmask, Meta_Data_membrane)
        
        voxel_size=reduce(lambda x, y: x * y, MetaData['scales ZYX'])
        EDcells_img=getImage(os.path.join(EDcells_folder_path,EDcellsMetaData['MetaData_EDcells']['EDcells file']))
        Volume=np.sum(EDcells_img>0)*voxel_size

        unique_values = np.unique(EDcells_img)
        N_objects = len(unique_values)-1

        data = {
            'data_name': data_name,
            'Volume ED': Volume,
            'N_objects': N_objects,
            'condition': MetaData.get('condition', None),
            'time in hpf': MetaData.get('time in hpf', None),
            'experimentalist': MetaData.get('experimentalist', None),
            'genotype': MetaData.get('genotype', None)
        }
        
        data_list.append(data)
        
        cell_df=getCellProps(EDcells_img,MetaData['scales ZYX'])
    


    df = pd.DataFrame(data_list)
    print(df)
    df.to_hdf(os.path.join(Curv_Thick_path,'scalarGrowthDataED.h5'), key='data', mode='w')
        

if __name__ == "__main__":
    main()