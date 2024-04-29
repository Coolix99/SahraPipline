import numpy as np
from typing import List
import napari
import os
import git
import pandas as pd
from simple_file_checksum import get_checksum
import pyvista as pv
from scipy import ndimage
from scipy.spatial import cKDTree

from config import *
from IO import *

def project(mesh:pv.PolyData,sig_img,scales):
    
    print(mesh.points.shape)

    pixel_pos = np.array(np.where(sig_img > 0)).T
    tree = cKDTree(mesh.points)
    distances, indices = tree.query(pixel_pos * scales, eps=0.2, k=1, workers=10, distance_upper_bound=5)
    Max = np.zeros(int(mesh.points.shape[0]), dtype=np.float32)
    valid_indices = indices < mesh.points.shape[0]
    np.maximum.at(Max, indices[valid_indices], sig_img[sig_img > 0][valid_indices])

    mesh.point_data['Max']=Max

    # print(np.max(Max))
    # print(np.mean(Max))
    # print(np.min(Max))

    # print(np.max(sig_img))
    # print(np.mean(sig_img))
    # print(np.min(sig_img))

    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, show_edges=False,scalars='Max',cmap='jet')
    # plotter.show()

    vertices = mesh.points/scales  # Nx3 array of vertex positions (x, y, z)
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]  # Mx3 array of vertex indices for each triangle

    # Start a Napari viewer
    viewer = napari.Viewer()
    viewer.add_image(sig_img)
    viewer.add_surface((vertices, faces,Max))

    # Start the Napari GUI loop
    napari.run()

    return Max
    

def evalStatus_proj(fin_dir_path):
    MetaData=get_JSON(fin_dir_path)
    if not 'Mesh_MetaData' in MetaData:
        print('no Mesh_MetaData')
        return False
    
    if not 'project_MetaData' in MetaData:
        return MetaData

    if not MetaData['project_MetaData']['project version']==project_version:
        return MetaData  

    if not MetaData['project_MetaData']['input mesh checksum']==MetaData['Mesh_MetaData']['output mesh checksum']:
        return MetaData

    return False


def make_projections():
    signal_dir_path=os.path.join(EpFlat_path,'signal_upperlayer')
    fin_list=[folder for folder in os.listdir(EpSeg_path) if os.path.isdir(os.path.join(EpSeg_path, folder))]
    for fin_folder in fin_list:
        print(fin_folder)
    
        fin_dir_path=os.path.join(EpSeg_path,fin_folder)
        
        PastMetaData=evalStatus_proj(fin_dir_path)
        if not isinstance(PastMetaData,dict):
            continue

        signal_path=os.path.join(signal_dir_path,fin_folder+'.tif')
        sig_img=getImage(signal_path)
        
        MetaData_mesh=PastMetaData['Mesh_MetaData']
       
        scales=MetaData_mesh['scales'].copy()
        Mesh_file=MetaData_mesh['Mesh file']
        Mesh_file_path=os.path.join(fin_dir_path,Mesh_file)
        mesh=pv.read(Mesh_file_path)

        print('compute proj')
        max_proj=project(mesh,sig_img,scales)
        
        proj_file_name=fin_folder+'_proj'
        saveArr(max_proj,os.path.join(fin_dir_path,proj_file_name))
        
        MetaData_proj={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_proj['git hash']=sha
        MetaData_proj['git repo']='meshProjection'
        MetaData_proj['proj version']=project_version
        MetaData_proj['proj file']=proj_file_name
        MetaData_proj['scales']=MetaData_mesh['scales']
        MetaData_proj['condition']=MetaData_mesh['condition']
        MetaData_proj['time in hpf']=MetaData_mesh['time in hpf']
        MetaData_proj['experimentalist']='Sahra'
        MetaData_proj['genotype']='WT'
        MetaData_proj['input mesh checksum']=MetaData_mesh['output mesh checksum']
        check_proj=get_checksum(os.path.join(fin_dir_path,proj_file_name), algorithm="SHA1")
        MetaData_proj['output proj checksum']=check_proj
        writeJSON(fin_dir_path,'project_MetaData',MetaData_proj)       


if __name__ == "__main__":
    make_projections() 
