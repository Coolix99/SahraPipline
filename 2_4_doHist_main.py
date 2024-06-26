import pandas as pd
import networkx as nx
import pyvista as pv
from typing import List
import git
from simple_file_checksum import get_checksum
from scipy.spatial import cKDTree

from config import *
from IO import *

def check_folders(diffeos_df, sim_df):
    connections_in_diffeos = set(
    frozenset((row['init_folder'], row['target_folder'])) for _, row in diffeos_df.iterrows()
    )

    # Check each pair from sim_df
    all_connections_found = True
    missing_connections = []

    for _, row in sim_df.iterrows():
        connection = frozenset((row['Folder1'], row['Folder2']))
        if connection not in connections_in_diffeos:
            all_connections_found = False
            missing_connections.append((row['Folder1'], row['Folder2']))

    # Print results
    if all_connections_found:
        #print("All connections from sim_df are contained in diffeos_df.")
        return True
    else:
        print("Some connections from sim_df are not found in diffeos_df:")
        for conn in missing_connections:
            print(f"Missing connection: {conn[0]} - {conn[1]}")
        return False


def find_ref_fin(sim_df):
    G = nx.Graph()

    # Add edges from DataFrame
    for index, row in sim_df.iterrows():
        G.add_edge(row['Folder1'], row['Folder2'], weight=row['Distance'])

    # Calculate closeness centrality
    centrality = nx.closeness_centrality(G, distance='weight')
    
    # Find the node with the highest closeness centrality
    most_central_node = max(centrality, key=centrality.get)   
    print('use: ',most_central_node)

    paths = {node: nx.shortest_path(G, source=node, target=most_central_node, weight='weight')
         for node in G if node != most_central_node}
    

    return most_central_node,paths

def getHistSurface(mesh:pv.PolyData):
    current_vertex_count = mesh.n_points
    total_surface_area = mesh.area

    # Define the desired number of vertices per square unit (modify this value as needed)
    desired_density_per_unit_area = 0.01

    # Calculate the target number of vertices
    target_vertex_count = desired_density_per_unit_area * total_surface_area

    # Calculate the decimation factor
    if target_vertex_count < current_vertex_count:
        decimation_factor = target_vertex_count / current_vertex_count
    else:
        decimation_factor = 1.0  # No decimation needed if the target is higher than current

    # Print information
    # print(f"Current vertex count: {current_vertex_count}")
    # print(f"Total surface area: {total_surface_area:.2f} square units")
    # print(f"Desired density per unit area: {desired_density_per_unit_area} vertices/square unit")
    # print(f"Target vertex count: {int(target_vertex_count)}")
    # print(f"Decimation factor: {decimation_factor:.2f}")

    # If decimation is needed, perform it
    if decimation_factor < 1.0:
        simplified_mesh = mesh.decimate(1-decimation_factor)
        # print(f"Simplified vertex count: {simplified_mesh.n_points}")

    tree = cKDTree(mesh.points)
    min_distance_point, index_point = tree.query(simplified_mesh.points, k=1)
    simplified_mesh.point_data['coord_1']=mesh.point_data['coord_1'][index_point]
    simplified_mesh.point_data['coord_2']=mesh.point_data['coord_2'][index_point]

    return simplified_mesh

def check_order(df, init_folder, target_folder):
    for index, row in df.iterrows():
        if row['init_folder'] == init_folder and row['target_folder'] == target_folder:
            return 1,row['diffeo_array']
        elif row['init_folder'] == target_folder and row['target_folder'] == init_folder:
            return -1,row['diffeo_array']
    return 0,None

def InterpolatedCoordinate_back(current_positions,cell_index,pull_back_surface:pv.PolyData,orig_surface:pv.PolyData):
    for i in range(cell_index.shape[0]):
        pos_inside=current_positions[i]
        pull_back_coords=np.zeros((3,3))
        orig_coords=np.zeros((3,3))
        for k in range(3):
            indx=pull_back_surface.get_cell(cell_index[i]).GetPointId(k)
            pull_back_coords[k]=pull_back_surface.points[indx]
            orig_coords[k]=orig_surface.points[indx]
        A = np.vstack([pull_back_coords[:, :2].T, np.ones(3)])
        b = np.hstack([pos_inside[:2], 1])
        barycentric_coords = np.linalg.lstsq(A, b, rcond=None)[0]
        interpolated_vector = np.dot(barycentric_coords, orig_coords)
        current_positions[i]=interpolated_vector
        
def InterpolatedCoordinate_forward(current_positions,cell_index,diff_arr,orig_surface:pv.PolyData):
    for i in range(cell_index.shape[0]):
        pos_inside=current_positions[i]
        orig_coords=np.zeros((3,3))
        diff_coord=np.zeros((3,3))
        for k in range(3):
            indx=orig_surface.get_cell(cell_index[i]).GetPointId(k)
            orig_coords[k]=orig_surface.points[indx]
            diff_coord[k]=diff_arr[indx]

        A = np.vstack([orig_coords[:, :2].T, np.ones(3)])
        b = np.hstack([pos_inside[:2], 1])
        barycentric_coords = np.linalg.lstsq(A, b, rcond=None)[0]
        interpolated_vector = np.dot(barycentric_coords, diff_coord)
        current_positions[i]=interpolated_vector
        

def calcDiffeo(path,used_diffeos_df,all_surfaces:List[pv.PolyData]):
    start_surface=all_surfaces[path[0]]
    current_positions=start_surface.points
    
    for i in range(len(path)-1):
        init_folder=path[i]
        target_folder=path[i+1]
        order,diff_arr = check_order(used_diffeos_df, init_folder, target_folder)
        if order==1:
            cell_index,current_positions = all_surfaces[path[i]].find_closest_cell(current_positions,return_closest_point=True)
            InterpolatedCoordinate_forward(current_positions,cell_index,diff_arr,all_surfaces[path[i]])
            continue
        if order==-1:
            pull_back_surface:pv.PolyData=all_surfaces[path[i+1]].copy()
            pull_back_surface.points=diff_arr
            cell_index,current_positions = pull_back_surface.find_closest_cell(current_positions,return_closest_point=True)
            InterpolatedCoordinate_back(current_positions,cell_index,pull_back_surface,all_surfaces[path[i+1]])
            continue
        print('unexpected')
        print(used_diffeos_df)
        print(init_folder, target_folder)
        raise
    return current_positions

def make_Hist(sim_df,diffeos_df):
    # Execute the function
    if not check_folders(diffeos_df, sim_df):
        return None
    ref_fin, paths = find_ref_fin(sim_df)

    unique_folders = set(sim_df['Folder1']).union(set(sim_df['Folder2']))
    all_surfaces={}
    for folder in unique_folders:
        folder_path=os.path.join(FlatFin_path,folder)
        surface_file_name=get_JSON(folder_path)['Thickness_MetaData']['Surface file']
        mesh=pv.read(os.path.join(folder_path,surface_file_name))
        all_surfaces[folder]=mesh

    used_diffeos_df = diffeos_df[diffeos_df['init_folder'].isin(unique_folders) | diffeos_df['target_folder'].isin(unique_folders)]
    used_diffeos_df['diffeo_array'] = [None] * len(used_diffeos_df)

    for index, row in used_diffeos_df.iterrows():
        diff_folder_path=os.path.join(ElementaryDiffeos_path,row['diff_folder'])
        diff_file_name=get_JSON(diff_folder_path)['MetaData_Diffeo']['Diffeo file']
        used_diffeos_df.at[index, 'diffeo_array'] = np.load(os.path.join(diff_folder_path,diff_file_name))


    HistSurface=getHistSurface(all_surfaces[ref_fin])
    N=HistSurface.n_points

    Hist_categories=['avg_curvature','gauss_curvature','thickness']
    n_categories=len(Hist_categories)
    Hist_data=np.empty((N, n_categories), dtype=object) 
    for i in range(N):
        for j in range(n_categories):
            Hist_data[i, j] = []

    ref_points=all_surfaces[ref_fin].points
    for i in range(ref_points.shape[0]):
        hist_ind=HistSurface.find_closest_point(ref_points[i])
        for k in range(n_categories):
            Hist_data[hist_ind,k].append(all_surfaces[ref_fin].point_data[Hist_categories[k]][i])

    for start_fin in paths:
        print(paths[start_fin])
        res=calcDiffeo(paths[start_fin],used_diffeos_df,all_surfaces)
        # plotter = pv.Plotter()
        # plotter.add_mesh(all_surfaces[ref_fin], color='blue', label='Target')
        # all_surfaces[start_fin].points=res
        # plotter.add_mesh(all_surfaces[start_fin], color='red', label='Defomed')
        # plotter.add_legend()
        # plotter.show()
       
        for i in range(res.shape[0]): 
            hist_ind=HistSurface.find_closest_point(res[i])
            for k in range(n_categories):
                Hist_data[hist_ind,k].append(all_surfaces[start_fin].point_data[Hist_categories[k]][i])


    avg_data=np.zeros_like(Hist_data,dtype=float)
    std_data=np.zeros_like(Hist_data,dtype=float)
    for i in range(Hist_data.shape[0]):
        for j in range(Hist_data.shape[1]):
            avg_data[i,j]=np.average(Hist_data[i,j])
            std_data[i,j]=np.std(Hist_data[i,j])
    for i in range(n_categories):
        cat=Hist_categories[i]
        HistSurface.point_data[cat+'_avg']=avg_data[:,i]
        HistSurface.point_data[cat+'_std']=avg_data[:,i]

    #print(HistSurface.point_data)
    plotter = pv.Plotter()
    plotter.add_mesh(HistSurface, scalars='thickness_avg', cmap='viridis', show_scalar_bar=True)
    plotter.show()
    return HistSurface,Hist_data
    
def evalStatus_Hist(Hist_path_dir):
    return

def make_Hists(relevant_conditions,diffeos_df):
    relevant_conditions.sort()
    combined_string = '_'.join(relevant_conditions)
    MST_path_dir=os.path.join(SimilarityMST_path,combined_string)

    Condition_path_dir=os.path.join(Hist_path,combined_string)
    make_path(Condition_path_dir)

    #print(os.listdir(MST_path_dir))
    all_sim_dfs=pd.read_hdf(os.path.join(MST_path_dir,'alldf.h5'))
    for index, row in all_sim_dfs.iterrows():
        print(row)
        file_name=row['file name']
        data_name=file_name[:-len('.h5')]
        sim_df=pd.read_hdf(os.path.join(MST_path_dir,file_name))

        Hist_path_dir=os.path.join(Condition_path_dir,data_name)

        # PastMetaData=evalStatus_Hist(Hist_path_dir)
        # if not isinstance(PastMetaData,dict):
        #     continue

        make_path(Hist_path_dir)
        
        res=make_Hist(sim_df,diffeos_df)
        if res is None:
            continue
        HistSurface,Hist_data=res
        saveArr(Hist_data,os.path.join(Hist_path_dir,data_name+'_histdata'))
        HistSurface.save(os.path.join(Hist_path_dir,data_name+'_surface.vtk'))
        
        MetaData_Hist={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_Hist['git hash']=sha
        MetaData_Hist['git repo']='Sahrapipline'
        MetaData_Hist['Hist version']=Hist_version
        MetaData_Hist['Surface file']=data_name+'_surface.vtk'
        MetaData_Hist['Data file']=data_name+'_histdata'
        check_Surface=get_checksum(os.path.join(Hist_path_dir,data_name+'_surface.vtk'), algorithm="SHA1")
        MetaData_Hist['output Surface checksum']=check_Surface
        check_Data=get_checksum(os.path.join(Hist_path_dir,data_name+'_histdata'), algorithm="SHA1")
        MetaData_Hist['output histdata checksum']=check_Data
        writeJSON(Hist_path_dir,'Hist_MetaData',MetaData_Hist)

    

def main():
    diffeos_df = pd.read_hdf(os.path.join(ElementaryDiffeos_path,'alldf.h5'))
    
    make_Hists(['time in hpf','condition'],diffeos_df)
    make_Hists(['condition'],diffeos_df)
    make_Hists(['time in hpf'],diffeos_df)
    return
    
    pass

if __name__ == "__main__":
    main()