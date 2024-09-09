from typing import List
import pyvista as pv
import numpy as np
import pandas as pd
import git
import hashlib
from simple_file_checksum import get_checksum

from sub_manifold import sub_manifold,sub_manifold_1_closed,sub_manifold_0
from find_diffeo_hirarchical_correct import findDiffeo,check_for_singularity
from config import *
from IO import *

from boundary import getBoundary,path_surrounds_point


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
        init_surface_file_name=init_MetaData['Thickness_MetaData']['Surface file']
        init_orient_file_name=init_MetaData['Orient_MetaData']['Orient file']
    except:
        return False
    target_MetaData=get_JSON(target_dir_path)
    try:
        target_surface_file_name=target_MetaData['Thickness_MetaData']['Surface file']
        target_orient_file_name=target_MetaData['Orient_MetaData']['Orient file']
    except:
        return False
    
    scales_init=init_MetaData['CenterLine_MetaData']['scales ZYX'].copy()
    scales_target=target_MetaData['CenterLine_MetaData']['scales ZYX'].copy()
   
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

    return full_path,folder_name

def eval_status(init_dir_path,target_dir_path,diffeo_dir_path):
    init_MetaData=get_JSON(init_dir_path)
    if not 'Thickness_MetaData' in init_MetaData:
        print('no Thickness_MetaData init')
        return False
    target_MetaData=get_JSON(target_dir_path)
    if not 'Thickness_MetaData' in target_MetaData:
        print('no Thickness_MetaData target')
        return False

    Diffeo_MetaData=get_JSON(diffeo_dir_path)
    if not 'MetaData_Diffeo' in Diffeo_MetaData:
        return True

    if not Diffeo_MetaData['MetaData_Diffeo']['input init checksum']==init_MetaData['Thickness_MetaData']['output Surface checksum']:
        return True
    if not Diffeo_MetaData['MetaData_Diffeo']['input target checksum']==target_MetaData['Thickness_MetaData']['output Surface checksum']:
        return True
    if not Diffeo_MetaData['MetaData_Diffeo']['Diffeo version']==Diffeo_version:
        return True

    return False

def make_diffeo(f1,f2,group_path):
    init_folder=f1
    target_folder=f2
    #maybe swap
    init_dir_path=os.path.join(FlatFin_path,init_folder)
    target_dir_path=os.path.join(FlatFin_path,target_folder)
    diffeo_dir_path,diffeo_folder=generate_deterministic_folder_name(init_folder,target_folder,group_path)
    make_path(diffeo_dir_path)
    
    res=eval_status(init_dir_path,target_dir_path,diffeo_dir_path)
    if not res:
        print('skip')
        return
    
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
        diff_energy,a,b,c=findDiffeo(mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target)
        print(diff_energy)
        if not check_for_singularity(mesh_init.copy(),mesh_target):
            print('Sigular diffeo, do not safe it')
            return 
        
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
    MetaData_Diffeo['diff_energy']=diff_energy.item()
    MetaData_Diffeo['material coeff']=[a,b,c]
    MetaData_Diffeo['input init checksum']=init_MetaData['Thickness_MetaData']['output Surface checksum']
    MetaData_Diffeo['input target checksum']=target_MetaData['Thickness_MetaData']['output Surface checksum']
    check_Diffeo=get_checksum(diffeo_file, algorithm="SHA1")
    MetaData_Diffeo['output Diffeo checksum']=check_Diffeo
    #print(MetaData_Diffeo)
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
    

def main():
    filename = os.path.join(Diffeo_path,"primitive_similarity.h5")
    alldf = pd.read_hdf(filename)
    print(alldf)
    for index2, row2 in alldf.iterrows():
        print(row2)
        f1=row2['node_1']
        f2=row2['node_2']
        make_diffeo(f1,f2,ElementaryDiffeos_path)
        
                

if __name__ == "__main__":
    main()