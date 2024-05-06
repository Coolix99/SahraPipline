import numpy as np
#np.random.seed(42)
from typing import List
import napari
import os
import git
from simple_file_checksum import get_checksum
import pyvista as pv
import random
from scipy.spatial import cKDTree

from cellpose import models
from cellpose.core import run_net
from cellpose.models import Cellpose

from config import *
from IO import *


def euler_int(mesh:pv.PolyData,flow):
    vertices = mesh.points
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]
    viewer = napari.Viewer()
    #viewer.add_surface((vertices, faces, proj), name='Surface Mesh')
    n = 1
    sampled_points = vertices[::n]  # Reduce density for clarity
    sampled_flows = flow[:, ::n].T  # Transpose to align with napari's expected input
    vector_data = np.zeros((sampled_flows.shape[0], 2, 3))
    vector_data[:, 0, :] = sampled_points  # Start points
    vector_data[:, 1, :] =  sampled_flows *1  # End points, scale to adjust length
    viewer.add_vectors(vector_data, edge_width=0.1, edge_color='cyan',
                    name='Flow Vectors')
    napari.run()
    

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

    if not MetaData['seg_MetaData']['apply version']==apply_version:
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
        
        print('apply network')
        pos=euler_int(mesh,flow)
        return
        flow_file_name=fin_folder+'_flow'
        saveArr(flow,os.path.join(fin_dir_path,flow_file_name))
        
        MetaData_flow={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_flow['git hash']=sha
        MetaData_flow['git repo']='Sahrapipline'
        MetaData_flow['apply version']=apply_version
        MetaData_flow['flow file']=flow_file_name
        MetaData_flow['scales']=MetaData_mesh['scales']
        MetaData_flow['condition']=MetaData_mesh['condition']
        MetaData_flow['time in hpf']=MetaData_mesh['time in hpf']
        MetaData_flow['experimentalist']='Sahra'
        MetaData_flow['genotype']='WT'
        MetaData_flow['input mesh checksum']=MetaData_mesh['output mesh checksum']
        check_proj=get_checksum(os.path.join(fin_dir_path,flow_file_name), algorithm="SHA1")
        MetaData_flow['output flow checksum']=check_proj
        writeJSON(fin_dir_path,'apply_MetaData',MetaData_flow)       


if __name__ == "__main__":
    segment_all() 
