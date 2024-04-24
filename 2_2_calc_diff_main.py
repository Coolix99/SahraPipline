from typing import List
import pyvista as pv
import numpy as np
import pandas as pd

import git
import hashlib
from simple_file_checksum import get_checksum

from sub_manifold import sub_manifold,sub_manifold_1_closed,sub_manifold_0
from find_diffeo_hirarchical import findDiffeo
from config import *
from IO import *

from boundary import getBoundary,path_surrounds_point

def getExample():
    n = 20
    x = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
    y = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
    xx, yy = np.meshgrid(x, y)
    A, b = 100, 100
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

    # Get the points as a 2D NumPy array (N by 3)
    points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    #points[0:5, :]
    cloud = pv.PolyData(points)
    surf = cloud.delaunay_2d()
    mesh_init=surf
    indices = getBoundary(surf)
    sub_manifolds_init:List[sub_manifold]=[sub_manifold_1_closed(surf, 'boundary', indices)]

    ind = np.where((surf.points[:, 0] < -195) & (surf.points[:, 1] > 195))[0][0]
    sub_manifolds_init.append(sub_manifold_0(surf, 'P1', ind))
    ind = np.where((surf.points[:, 0] < -195) & (surf.points[:, 1] < -195))[0][0]
    sub_manifolds_init.append(sub_manifold_0(surf, 'P2', ind))
    ind = np.where((surf.points[:, 0] > 195) & (surf.points[:, 1] > 195))[0][0]
    sub_manifolds_init.append(sub_manifold_0(surf, 'P3', ind))
    ind = np.where((surf.points[:, 0] > 195) & (surf.points[:, 1] < -195))[0][0]
    sub_manifolds_init.append(sub_manifold_0(surf, 'P4', ind))
    
    n = 20
    x = np.linspace(-250, 200, num=n) + np.random.uniform(-5, 5, size=n)
    y = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
    xx, yy = np.meshgrid(x, y)
    A, b = 20, 100
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

    # Get the points as a 2D NumPy array (N by 3)
    points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    #points[0:5, :]
    cloud = pv.PolyData(points)
    surf = cloud.delaunay_2d()
    indices = getBoundary(surf)
    mesh_target=surf
    sub_manifolds_target:List[sub_manifold]=[sub_manifold_1_closed(surf, 'boundary', indices)]
    ind = np.where((surf.points[:, 0] < -195) & (surf.points[:, 1] > 195))[0][0]
    sub_manifolds_target.append(sub_manifold_0(surf, 'P1', ind))
    ind = np.where((surf.points[:, 0] < -195) & (surf.points[:, 1] < -195))[0][0]
    sub_manifolds_target.append(sub_manifold_0(surf, 'P2', ind))
    ind = np.where((surf.points[:, 0] > 195) & (surf.points[:, 1] > 195))[0][0]
    sub_manifolds_target.append(sub_manifold_0(surf, 'P3', ind))
    ind = np.where((surf.points[:, 0] > 195) & (surf.points[:, 1] < -195))[0][0]
    sub_manifolds_target.append(sub_manifold_0(surf, 'P4', ind))

    mesh_init.point_data["deformed"]=mesh_init.points.copy()
    mesh_init.point_data["displacement"]=mesh_init.point_data["deformed"]-mesh_init.points

    return mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target

def find_transformation_matrix(v1, v2, v3, u1, u2, u3):
    """Find the transformation matrix that maps v vectors to u vectors."""
    # Concatenate the v and u vectors into a single matrix
    A = np.vstack((v1, v2, v3)).T
    B = np.vstack((u1, u2, u3)).T
    transformation_matrix=B@np.linalg.inv(A)
    return transformation_matrix

def transformation_matrix(v1, v2, v3, u1, u2, u3, center_point1, center_point2):
    """Find the transformation matrix that maps unit_vector1 to unit_vector2
    and center_point1 to center_point2."""
    rotation_matrix = find_transformation_matrix(v1, v2, v3, u1, u2, u3)
    translation_vector = center_point2 - np.dot(rotation_matrix, center_point1)

    return rotation_matrix,translation_vector

def extract_coordinate(df, name):
    row = df[df['name'] == name]
    if not row.empty:
        return row['coordinate_px'].iloc[0]
    else:
        return None


def getInput(init_dir_path,target_dir_path):
    init_MetaData=get_JSON(init_dir_path)
    try:
        init_surface_file_name=init_MetaData['Coord_MetaData']['Surface file']
        init_orient_file_name=init_MetaData['Orient_MetaData']['Orient file']
    except:
        return False
    target_MetaData=get_JSON(target_dir_path)
    try:
        target_surface_file_name=target_MetaData['Coord_MetaData']['Surface file']
        target_orient_file_name=target_MetaData['Orient_MetaData']['Orient file']
    except:
        return False
    scales_init=init_MetaData['Coord_MetaData']['XYZ size in mum'].copy()
    if init_MetaData['Coord_MetaData']['axes']=='ZYX':
        scales_init[0], scales_init[-1] = scales_init[-1], scales_init[0]
    
    scales_target=target_MetaData['Coord_MetaData']['XYZ size in mum'].copy()
    if target_MetaData['Coord_MetaData']['axes']=='ZYX':
        scales_target[0], scales_target[-1] = scales_target[-1], scales_target[0]


    init_surface_file=os.path.join(init_dir_path,init_surface_file_name)
    init_orient_file=os.path.join(init_dir_path,init_orient_file_name)
    target_surface_file=os.path.join(target_dir_path,target_surface_file_name)
    target_orient_file=os.path.join(target_dir_path,target_orient_file_name)

    df_init_orient = pd.read_hdf(init_orient_file)
    df_target_orient = pd.read_hdf(target_orient_file)
    mesh_init = pv.read(init_surface_file)
    mesh_target = pv.read(target_surface_file)

    #print(df_init_orient)
    #print(df_target_orient)
    # print(mesh_init)
    # print(mesh_target)
    v1_init=(extract_coordinate(df_init_orient,'Proximal_pt')-extract_coordinate(df_init_orient,'Distal_pt'))*scales_init
    v2_init=(extract_coordinate(df_init_orient,'Anterior_pt')-extract_coordinate(df_init_orient,'Posterior_pt'))*scales_init
    v3_init=(extract_coordinate(df_init_orient,'Proximal2_pt')-extract_coordinate(df_init_orient,'Distal2_pt'))*scales_init
    n_init=np.cross(v3_init,v2_init)
    n_init = n_init / np.linalg.norm(n_init)
    v1_init = v1_init - n_init*np.dot(n_init,v1_init)
    v1_init = v1_init / np.linalg.norm(v1_init)
    v4_init = np.cross(n_init,v1_init)
    
    v1_target=(extract_coordinate(df_target_orient,'Proximal_pt')-extract_coordinate(df_target_orient,'Distal_pt'))*scales_target
    v2_target=(extract_coordinate(df_target_orient,'Anterior_pt')-extract_coordinate(df_target_orient,'Posterior_pt'))*scales_target
    v3_target=(extract_coordinate(df_target_orient,'Proximal2_pt')-extract_coordinate(df_target_orient,'Distal2_pt'))*scales_target
    n_target=np.cross(v3_target,v2_target)
    n_target = n_target / np.linalg.norm(n_target)
    v1_target = v1_target - n_target*np.dot(n_target,v1_target)
    v1_target = v1_target / np.linalg.norm(v1_target)
    v4_target = np.cross(n_target,v1_target)
    mesh_target.n_target=n_target

    center_init=mesh_init.center
    center_target=mesh_target.center

    rotation_matrix,translation_vector = transformation_matrix(v1_init,n_init,v4_init, v1_target,n_target,v4_target, center_init, center_target)
    # print(rotation_matrix@center_init+translation_vector)
    # print(rotation_matrix@v1_init)
    # print(rotation_matrix@n_init)
    # print(rotation_matrix@v4_init)
    mesh_init.point_data['rotated']=(rotation_matrix@mesh_init.points.T+translation_vector[:,np.newaxis]).T
   
    #calculate boundary
    #make submanifolds
    indices_init = getBoundary(mesh_init)
    if path_surrounds_point(mesh_init.point_data['rotated'][indices_init],center_target,n_target)=='Counterclockwise':
        indices_init=indices_init[::-1]
    sub_manifolds_init:List[sub_manifold]=[sub_manifold_1_closed(mesh_init, 'boundary', indices_init)]

    indices_target = getBoundary(mesh_target)
    if path_surrounds_point(mesh_target.points[indices_target],center_target,n_target)=='Counterclockwise':
        indices_target=indices_target[::-1]
    sub_manifolds_target:List[sub_manifold]=[sub_manifold_1_closed(mesh_target, 'boundary', indices_target)]

    return mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target,init_MetaData,target_MetaData

def getDeformedMesh(mesh):
    deformed_mesh=mesh.copy()
    deformed_mesh.points=deformed_mesh.point_data["deformed"]
    return deformed_mesh

def generate_deterministic_folder_name(input1, input2, base_path="."):
    # Combine the inputs and hash them
    combined_input = f"{input1}_{input2}".encode('utf-8')
    hash_output = hashlib.sha256(combined_input).hexdigest()[:10]  # Taking only first 10 characters for brevity

    # Generate a folder name based on the hash
    folder_name = f"diffeo_{hash_output}"
    full_path = os.path.join(base_path, folder_name)

    # Check if folder exists and generate a new one if necessary
    # counter = 1
    # while os.path.exists(full_path):
    #     new_combined_input = f"{combined_input}_{counter}".encode('utf-8')
    #     new_hash_output = hashlib.sha256(new_combined_input).hexdigest()[:10]
    #     folder_name = f"diffeo_{new_hash_output}"
    #     full_path = os.path.join(base_path, folder_name)
    #     counter += 1

    return full_path,folder_name

def eval_status(init_dir_path,target_dir_path,diffeo_dir_path):
    init_MetaData=get_JSON(init_dir_path)
    if not 'Coord_MetaData' in init_MetaData:
        print('no coord init')
        return False
    target_MetaData=get_JSON(target_dir_path)
    if not 'Coord_MetaData' in target_MetaData:
        print('no coord target')
        return False

    Diffeo_MetaData=get_JSON(diffeo_dir_path)
    if not 'MetaData_Diffeo' in Diffeo_MetaData:
        return True

    if not Diffeo_MetaData['MetaData_Diffeo']['input init checksum']==init_MetaData['Coord_MetaData']['output Surface checksum']:
        return True
    if not Diffeo_MetaData['MetaData_Diffeo']['input target checksum']==target_MetaData['Coord_MetaData']['output Surface checksum']:
        return True
    if not Diffeo_MetaData['MetaData_Diffeo']['Diffeo version']==Diffeo_version:
        return True

    return False

def make_diffeo(f1,f2,group_path):
    init_folder=f1
    target_folder=f2
    #maybe swap
    init_dir_path=os.path.join(LMcoord_path,init_folder)
    target_dir_path=os.path.join(LMcoord_path,target_folder)
    diffeo_dir_path,diffeo_folder=generate_deterministic_folder_name(init_folder,target_folder,group_path)
    make_path(diffeo_dir_path)
    
    res=eval_status(init_dir_path,target_dir_path,diffeo_dir_path)
    if not res:
        print('skip')
        return

    #mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target=getExample()
    mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target,init_MetaData,target_MetaData=getInput(init_dir_path,target_dir_path)
    writeJSON(diffeo_dir_path,'init_MetaData',init_MetaData)
    writeJSON(diffeo_dir_path,'target_MetaData',target_MetaData)
    
    #with rotation
    # p = pv.Plotter()
    # delta=0
    # mesh_init.points=mesh_init.point_data['rotated']-np.array((0,0,delta))
    # p.add_mesh(mesh_init, color='blue', show_edges=True)
    # points_array=mesh_init.point_data['rotated'][sub_manifolds_init[0].path]-np.array((0,0,delta))
    # polydata = pv.PolyData(points_array)
    # lines=np.zeros((len(points_array)+1),dtype=int)
    # lines[0]=len(points_array)
    # lines[1:] = np.arange(len(points_array),dtype=int)
    # polydata.lines = lines
    # p.add_mesh(polydata, color='blue', line_width=5, render_lines_as_tubes=True)  
    
    # p.add_mesh(mesh_target, color='green', show_edges=True)
    # points_array=mesh_target.points[sub_manifolds_target[0].path]
    # polydata = pv.PolyData(points_array)
    # lines=np.zeros((len(points_array)+1),dtype=int)
    # lines[0]=len(points_array)
    # lines[1:] = np.arange(len(points_array),dtype=int)
    # polydata.lines = lines
    # p.add_mesh(polydata, color='green', line_width=5, render_lines_as_tubes=True)  
    # p.show()
    

    #make_path(diffeo_dir_path)
    print('calculate',init_folder,target_folder)
    try:
        findDiffeo(mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target)
    except:
        print('fatal error')
        return

    #mesh_deformed=getDeformedMesh(mesh_init)
    #mesh_deformed.clear_point_data()
    
    deformed_positions=mesh_init.point_data["deformed"]

    diffeo_file_name=diffeo_folder+'_deformed_positions.npy'
    diffeo_file=os.path.join(diffeo_dir_path,diffeo_file_name)
    saveArr(deformed_positions,diffeo_file)

    MetaData_Diffeo={}
    repo = git.Repo(gitPath,search_parent_directories=True)
    sha = repo.head.object.hexsha
    MetaData_Diffeo['git hash']=sha
    MetaData_Diffeo['git repo']='FinDiffeo'
    MetaData_Diffeo['Diffeo version']=Diffeo_version
    MetaData_Diffeo['Diffeo file']=diffeo_file_name
    MetaData_Diffeo['init_folder']=init_folder
    MetaData_Diffeo['target_folder']=target_folder
    MetaData_Diffeo['input init checksum']=init_MetaData['Coord_MetaData']['output Surface checksum']
    MetaData_Diffeo['input target checksum']=target_MetaData['Coord_MetaData']['output Surface checksum']
    check_Diffeo=get_checksum(diffeo_file, algorithm="SHA1")
    MetaData_Diffeo['output Diffeo checksum']=check_Diffeo
    writeJSON(diffeo_dir_path,'MetaData_Diffeo',MetaData_Diffeo)
    
    # p = pv.Plotter()
    # p.add_mesh(mesh_target, color='green', show_edges=True)
    # points_array=mesh_target.points[sub_manifolds_target[0].path]
    # polydata = pv.PolyData(points_array)
    # lines=np.zeros((len(points_array)+1),dtype=int)
    # lines[0]=len(points_array)
    # lines[1:] = np.arange(len(points_array),dtype=int)
    # polydata.lines = lines
    # p.add_mesh(polydata, color='green', line_width=5, render_lines_as_tubes=True)  
    # delta=30
    # mesh_init.points=mesh_init.point_data['deformed']-np.array((delta,0,0))
    # p.add_mesh(mesh_init, color='blue', show_edges=True)
    # points_array=mesh_init.point_data['deformed'][sub_manifolds_init[0].path]-np.array((delta,0,0))
    # polydata = pv.PolyData(points_array)
    # lines=np.zeros((len(points_array)+1),dtype=int)
    # lines[0]=len(points_array)
    # lines[1:] = np.arange(len(points_array),dtype=int)
    # polydata.lines = lines
    # p.add_mesh(polydata, color='blue', line_width=5, render_lines_as_tubes=True)  
    # p.show()

    # #init (without rotation)
    # p = pv.Plotter()
    # p.add_mesh(mesh_init, color='blue', show_edges=True)
    # #points_cloud = pv.PolyData(mesh_init.points[sub_manifolds_init[0].path])
    # #p.add_points(points_cloud, color="green", point_size=10)
    # points_array=mesh_init.points[sub_manifolds_init[0].path]
    # polydata = pv.PolyData(points_array)
    # lines=np.zeros((len(points_array)+1),dtype=int)
    # lines[0]=len(points_array)
    # lines[1:] = np.arange(len(points_array),dtype=int)
    # polydata.lines = lines
    # p.add_mesh(polydata, color='blue', line_width=5, render_lines_as_tubes=True)  
    
    # p.add_mesh(mesh_target, color='green', show_edges=True)
    # delta=0
    # points_array=mesh_target.points[sub_manifolds_target[0].path]
    # polydata = pv.PolyData(points_array)
    # lines=np.zeros((len(points_array)+1),dtype=int)
    # lines[0]=len(points_array)
    # lines[1:] = np.arange(len(points_array),dtype=int)
    # polydata.lines = lines
    # p.add_mesh(polydata, color='green', line_width=5, render_lines_as_tubes=True)  
    # p.show()
    
    # #with rotation
    # p = pv.Plotter()
    # delta=0
    # mesh_init.points=mesh_init.point_data['rotated']-np.array((0,0,delta))
    # p.add_mesh(mesh_init, color='blue', show_edges=True)
    # points_array=mesh_init.point_data['rotated'][sub_manifolds_init[0].path]-np.array((0,0,delta))
    # polydata = pv.PolyData(points_array)
    # lines=np.zeros((len(points_array)+1),dtype=int)
    # lines[0]=len(points_array)
    # lines[1:] = np.arange(len(points_array),dtype=int)
    # polydata.lines = lines
    # p.add_mesh(polydata, color='blue', line_width=5, render_lines_as_tubes=True)  
    
    # p.add_mesh(mesh_target, color='green', show_edges=True)
    # points_array=mesh_target.points[sub_manifolds_target[0].path]
    # polydata = pv.PolyData(points_array)
    # lines=np.zeros((len(points_array)+1),dtype=int)
    # lines[0]=len(points_array)
    # lines[1:] = np.arange(len(points_array),dtype=int)
    # polydata.lines = lines
    # p.add_mesh(polydata, color='green', line_width=5, render_lines_as_tubes=True)  
    # p.show()
    pass

def main():
    SimilarityMST_folder_list=[folder for folder in os.listdir(SimilarityMST_path) if os.path.isdir(os.path.join(SimilarityMST_path, folder))]
    for S_folder in SimilarityMST_folder_list:
        print(S_folder)
        make_path(os.path.join(ElementaryDiffeos_path,S_folder))
        filename = os.path.join(SimilarityMST_path,S_folder,"alldf.h5")
        alldf = pd.read_hdf(filename)
        print(alldf)
        for index, row in alldf.iterrows():
            print(row)
            file_name=row['file name']
            MST_df = pd.read_hdf(os.path.join(SimilarityMST_path,S_folder,file_name))
            group_path=os.path.join(ElementaryDiffeos_path,S_folder,file_name[:-len('.h5')])
            make_path(group_path)
            for index2, row2 in MST_df.iterrows():
                print(row2)
                f1=row2['Folder1']
                f2=row2['Folder2']
                make_diffeo(f1,f2,group_path)
        
    return

import pyvista as pv
import networkx as nx
import numpy as np

def getMesh():
    n = 20
    x = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
    y = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
    xx, yy = np.meshgrid(x, y)
    A, b = 100, 100
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

    # Get the points as a 2D NumPy array (N by 3)
    points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    points[0:5, :]
    cloud = pv.PolyData(points)
    surf = cloud.delaunay_2d()
    return surf

def is_edge_ignored(edge, edges_to_ignore):
    # Check if either direct or reversed edge is in the list
    for ignore_edge in edges_to_ignore:
        if np.array_equal(edge, ignore_edge) or np.array_equal(edge[::-1], ignore_edge):
            return True
    return False

def find_shared_edge(edges_1,edges_2):
    return list(set(edges_1) & set(edges_2))

def test():
    # Assuming `mesh` is your PyVista mesh and `edges_to_ignore` is a Nx2 numpy array of vertex indices
    mesh = getMesh()
    from find_diffeo import pyvista_mesh_to_networkx
    K = pyvista_mesh_to_networkx(mesh)
    shortest_path = nx.shortest_path(K, source=0, target=50, weight='weight')
    #print(shortest_path)
    edges_to_ignore = [[shortest_path[i], shortest_path[i+1]] for i in range(len(shortest_path)-1)]
    print(edges_to_ignore)

    # Create an empty graph
    G = nx.Graph()
    
    # Iterate through each cell in the mesh
    for ind in range(mesh.n_cells):
        G.add_node(ind)  # Add each cell as a node
        neighbors = mesh.cell_neighbors(ind, 'edges')
        # Iterate through neighbors to check for ignored edges
        for neighbor in neighbors:
            # Find the vertices that define the shared edge between ind and neighbor
            # This is a placeholder: you'll need to adapt this part to accurately identify shared edges between cells
            shared_edge = find_shared_edge(mesh.get_cell(ind).point_ids,mesh.get_cell(neighbor).point_ids)
            
            # Check if this edge should be ignored
            if not is_edge_ignored(shared_edge, edges_to_ignore):
                G.add_edge(ind, neighbor)
    return
if __name__ == "__main__":
    #test()
    main()