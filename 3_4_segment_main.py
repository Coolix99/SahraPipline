import numpy as np
#np.random.seed(42)
from typing import List
import napari
import os
import git
from simple_file_checksum import get_checksum
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from config import *
from IO import *


def euler_int(mesh:pv.PolyData,flow):


    N_iter = 10  
    dx = 5
    pos=mesh.points.copy()
    tree = cKDTree(mesh.points)
    actual_flows=flow.copy()
    for i in range(N_iter):
        print(i)
        pos=pos+dx*actual_flows.T
        _, indices = tree.query(pos)
        pos = mesh.points[indices]
        actual_flows=flow[:,indices]

    # vertices = mesh.points
    # faces = mesh.faces.reshape((-1, 4))[:, 1:4]
    # viewer = napari.Viewer()
    # #viewer.add_surface((vertices, faces, proj), name='Surface Mesh')
    # n = 1
    # sampled_points = vertices[::n]  # Reduce density for clarity
    # sampled_flows = flow[:, ::n].T  # Transpose to align with napari's expected input
    # vector_data = np.zeros((sampled_flows.shape[0], 2, 3))
    # vector_data[:, 0, :] = sampled_points  # Start points
    # vector_data[:, 1, :] =  sampled_flows *1  # End points, scale to adjust length
    # viewer.add_vectors(vector_data, edge_width=0.1, edge_color='cyan',
    #                 name='Flow Vectors')
    # viewer.add_points(pos, size=1.0, face_color='red', name='Final Positions')
    # napari.run()

    return pos
    
def segCells(pos):
    clustering = DBSCAN(eps=5, min_samples=10).fit(pos)
    labels = clustering.labels_
    return labels

    


def evalStatus_seg(fin_dir_path):
    MetaData=get_JSON(fin_dir_path)
    if not 'Mesh_MetaData' in MetaData:
        print('no Mesh_MetaData')
        return False
    
    if not 'project_MetaData' in MetaData:
        print('no project_MetaData')
        return False
    
    if not 'apply_MetaData' in MetaData:
        print('no apply_MetaData')
        return False
    
    if not 'seg_MetaData' in MetaData:
        return MetaData

    if not MetaData['seg_MetaData']['seg version']==seg_version:
        return MetaData  

    if not MetaData['seg_MetaData']['input flow checksum']==MetaData['apply_MetaData']['output flow checksum']:
        return MetaData

    return False

def segment_all():  
    fin_list=[folder for folder in os.listdir(EpSeg_path) if os.path.isdir(os.path.join(EpSeg_path, folder))]
    for fin_folder in fin_list:
        print(fin_folder)
    
        fin_dir_path=os.path.join(EpSeg_path,fin_folder)
        
        PastMetaData=evalStatus_seg(fin_dir_path)
        if not isinstance(PastMetaData,dict):
            continue

        MetaData_mesh=PastMetaData['Mesh_MetaData']
        Mesh_file=MetaData_mesh['Mesh file']
        Mesh_file_path=os.path.join(fin_dir_path,Mesh_file)
        mesh=pv.read(Mesh_file_path)

        MetaData_apply=PastMetaData['apply_MetaData']
        flow_file=MetaData_apply['flow file']
        flow_file_path=os.path.join(fin_dir_path,flow_file)
        flow=loadArr(flow_file_path)
        
        print('euler int')
        pos=euler_int(mesh,flow)
        print(pos.shape)

        labels=segCells(pos)
        print(labels.shape)

        import matplotlib.pyplot as plt
        unique_integers = np.unique(labels)
        colors = plt.cm.get_cmap('nipy_spectral', len(unique_integers))
        np.random.seed(42) 
        indices = np.arange(len(unique_integers))
        np.random.shuffle(indices)
        color_mapping = {int_val: colors(idx / len(unique_integers)) for idx, int_val in zip(indices, unique_integers)}
        vertex_colors = np.array([color_mapping[val] for val in labels])
        viewer = napari.Viewer()
        vertices = mesh.points
        faces = mesh.faces.reshape((-1, 4))[:, 1:4]
        viewer.add_surface((vertices, faces),vertex_colors=vertex_colors, name='Surface Mesh')
        napari.run()

        
        seg_file_name=fin_folder+'_seg'
        saveArr(labels,os.path.join(fin_dir_path,seg_file_name))
        
        MetaData_seg={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_seg['git hash']=sha
        MetaData_seg['git repo']='Sahrapipline'
        MetaData_seg['seg version']=seg_version
        MetaData_seg['seg file']=seg_file_name
        MetaData_seg['scales']=MetaData_mesh['scales']
        MetaData_seg['condition']=MetaData_mesh['condition']
        MetaData_seg['time in hpf']=MetaData_mesh['time in hpf']
        MetaData_seg['experimentalist']='Sahra'
        MetaData_seg['genotype']='WT'
        MetaData_seg['input flow checksum']=MetaData_apply['output flow checksum']
        check_proj=get_checksum(os.path.join(fin_dir_path,seg_file_name), algorithm="SHA1")
        MetaData_seg['output seg checksum']=check_proj
        writeJSON(fin_dir_path,'seg_MetaData',MetaData_seg)       


if __name__ == "__main__":
    segment_all() 
