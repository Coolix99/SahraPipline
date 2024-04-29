import numpy as np
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

def compute_curve_patch(model:Cellpose,img):
    #todo: normalise and so on
    masks, flows, _=model.cp.eval(img)
    #res=run_net(model,img)
    print(masks.shape)
    print(len(flows))

def calculate_flow(mesh:pv.PolyData,proj,model):
    bsize=224
    im_scale=1

    num_points = mesh.points.shape[0]
    random_index = random.randint(0, num_points - 1)

    r0=mesh.points[random_index]
    n=mesh.point_normals[random_index]
    e1=np.array((1,0,0))
    e1=e1-np.dot(e1,n)
    e1=e1/np.linalg.norm(e1)
    e2=np.cross(e1,n)

    indices = np.arange(-bsize//2, bsize//2 )

    x = np.tile(indices, (bsize, 1))
    y = np.tile(indices[:, np.newaxis], (1, bsize))
    pos=x[:,:,np.newaxis]*e1[np.newaxis,np.newaxis,:]*im_scale+y[:,:,np.newaxis]*e2[np.newaxis,np.newaxis,:]*im_scale+r0[np.newaxis,np.newaxis,:]
    print(pos.shape)
    flattened_pos = pos.reshape(-1, 3)
    print(flattened_pos.shape)
    tree = cKDTree(mesh.points)
    _, indices = tree.query(flattened_pos)
    print(indices.shape)
    point_data = proj
    image_data = point_data[indices]
    print(image_data.shape)
    image_patch = image_data.reshape(bsize, bsize)

    # import matplotlib.pyplot as plt
    # plt.imshow(image_patch, cmap='viridis')  # You can change the colormap as needed
    # plt.colorbar()  # Optionally add a color bar to show the mapping of color to data values
    # plt.title('Image from Mesh Point Data')
    # plt.show()

    compute_curve_patch(model,image_patch)

    return


    vertices = mesh.points 
    faces = mesh.faces.reshape((-1, 4))[:, 1:4] 
    viewer = napari.Viewer()
    viewer.add_surface((vertices, faces,proj))
    napari.run()

    return 0
    

def evalStatus_apply(fin_dir_path):
    MetaData=get_JSON(fin_dir_path)
    if not 'Mesh_MetaData' in MetaData:
        print('no Mesh_MetaData')
        return False
    
    if not 'project_MetaData' in MetaData:
        print('no project_MetaData')
        return False
    
    if not 'apply_MetaData' in MetaData:
        return MetaData

    if not MetaData['apply_MetaData']['apply version']==apply_version:
        return MetaData  

    if not MetaData['apply_MetaData']['input mesh checksum']==MetaData['Mesh_MetaData']['output mesh checksum']:
        return MetaData
    
    if not MetaData['apply_MetaData']['input proj checksum']==MetaData['project_MetaData']['output proj checksum']:
        return MetaData

    return False

def apply_models():
    model = models.Cellpose(model_type='cyto2')
    
    fin_list=[folder for folder in os.listdir(EpSeg_path) if os.path.isdir(os.path.join(EpSeg_path, folder))]
    for fin_folder in fin_list:
        print(fin_folder)
    
        fin_dir_path=os.path.join(EpSeg_path,fin_folder)
        
        PastMetaData=evalStatus_apply(fin_dir_path)
        if not isinstance(PastMetaData,dict):
            continue

        MetaData_mesh=PastMetaData['Mesh_MetaData']
        #scales=MetaData_mesh['scales'].copy()
        Mesh_file=MetaData_mesh['Mesh file']
        Mesh_file_path=os.path.join(fin_dir_path,Mesh_file)
        mesh=pv.read(Mesh_file_path)

        MetaData_proj=PastMetaData['project_MetaData']
        proj_file=MetaData_proj['proj file']
        proj_file_path=os.path.join(fin_dir_path,proj_file)
        proj=loadArr(proj_file_path)

        print('apply network')
        flow=calculate_flow(mesh,proj,model)
        return
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
    apply_models() 
