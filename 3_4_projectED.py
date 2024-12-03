import pandas as pd
import git
from simple_file_checksum import get_checksum
import pyvista as pv
from scipy.spatial import cKDTree

from config import *
from IO import *

def find_closest_points(df_prop, mesh:pv.PolyData):
    # Extract scaled centroids
    centroids_scaled = df_prop[['centroids_scaled Z', 'centroids_scaled Y', 'centroids_scaled X']].to_numpy()

    # Build KDTree for mesh points
    mesh_points = mesh.points
    kdtree_mesh = cKDTree(mesh_points)

    # Placeholder for results
    results = []

    for idx, centroid in enumerate(centroids_scaled):
        # Find closest point in mesh
        _, closest_idx_mesh = kdtree_mesh.query(centroid)
        closest_point_mesh = mesh_points[closest_idx_mesh]

        _,closest_Point=mesh.find_closest_cell(centroid,return_closest_point=True)
        dist_mesh=np.linalg.norm(closest_Point-centroid)

        # Add results
        results.append({
            'Label': int(df_prop.iloc[idx]['Label']),
            'Closest Mesh Point Index': closest_idx_mesh,
            'Closest Mesh Point Z': closest_point_mesh[0],
            'Closest Mesh Point Y': closest_point_mesh[1],
            'Closest Mesh Point X': closest_point_mesh[2],
            'Closest  Point Z': closest_Point[0],
            'Closest  Point Y': closest_Point[1],
            'Closest  Point X': closest_Point[2],
            'Distance to Mesh': dist_mesh,
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)


    return results_df

def main():
    EDprops_folder_list= [item for item in os.listdir(ED_cell_props_path) if os.path.isdir(os.path.join(ED_cell_props_path, item))]
    for EDprop_folder in EDprops_folder_list:
        print(EDprop_folder)
        EDprop_folder_path=os.path.join(ED_cell_props_path,EDprop_folder)
        
        EDpropMetaData=get_JSON(EDprop_folder_path)
        if EDpropMetaData=={}:
            print('no EDprops')
            continue
        df_prop = pd.read_hdf(os.path.join(EDprop_folder_path,EDpropMetaData['MetaData_EDcell_props']['EDcells file']), key='data')

        folder_path=os.path.join(FlatFin_path,EDprop_folder+'_FlatFin')
        FF_MetaData=get_JSON(folder_path)
        if FF_MetaData=={}:
            print('no FF_MetaData')
            continue
        surface_file_name=FF_MetaData['Thickness_MetaData']['Surface file']
        mesh=pv.read(os.path.join(folder_path,surface_file_name))

        # print(df_prop)
        # print(mesh)

        df_proj=find_closest_points(df_prop, mesh)
        #print(df_proj)
        df_proj.to_hdf(os.path.join(EDprop_folder_path,'cell_proj.h5'), key='data', mode='w')
        
        MetaData_EDcell_proj={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_EDcell_proj['git hash']=sha
        MetaData_EDcell_proj['git repo']='Sahrapipline'
        MetaData_EDcell_proj['EDcell_proj file']='cell_proj.h5'
        MetaData_EDcell_proj['condition']=EDpropMetaData['MetaData_EDcell_top']['condition']
        MetaData_EDcell_proj['time in hpf']=EDpropMetaData['MetaData_EDcell_top']['time in hpf']
        MetaData_EDcell_proj['genotype']=EDpropMetaData['MetaData_EDcell_top']['genotype']
        MetaData_EDcell_proj['scales ZYX']=EDpropMetaData['MetaData_EDcell_top']['scales ZYX']
        check=get_checksum(os.path.join(EDprop_folder_path,'cell_props.h5'), algorithm="SHA1")
        MetaData_EDcell_proj['EDcell_proj checksum']=check
        writeJSON(EDprop_folder_path,'MetaData_EDcell_proj',MetaData_EDcell_proj)       




if __name__ == "__main__":
    main()