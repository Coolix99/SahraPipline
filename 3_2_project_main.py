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

def project(signal_file, vol_file, mesh_file,scales,ending):
    signal = getImage(signal_file)
    vol=getImage(vol_file)
    mesh=pv.read(mesh_file)
    subdivided_mesh = mesh.subdivide(3)
    print(mesh.points.shape)
    print(subdivided_mesh.points.shape)

    tree = cKDTree(subdivided_mesh.points)
    distances, indices = tree.query(mesh.points,eps=0.2, k=1,workers=10)


    print(signal[vol].shape)
    pixel_pos=np.array(np.where(vol)).T
    print(pixel_pos.shape)

    tree = cKDTree(subdivided_mesh.points)
    distances, indices = tree.query(pixel_pos*scales,eps=0.2, k=1,workers=10)
    print(distances.shape)
    print(indices.shape)

    Max=np.zeros(int(subdivided_mesh.points.shape[0]),dtype=np.float32)
    try:
        np.maximum.at(Max,indices,signal[vol])
    except:
        return None
    subdivided_mesh.point_data[ending]=Max
    return subdivided_mesh
    #return subdivided_mesh
    # plotter = pv.Plotter()
    # plotter.add_mesh(subdivided_mesh, show_edges=True,scalars=ending,cmap='jet')
    # plotter.show()

def evalStatus_proj(signal_dir_path,vol_dir_path,LMcoord_dir_path,project_dir_path,ending):
    """
    checks MetaData to desire wether to evaluate the file or not
    returns a dict of MetaData to be written if we should evaluate the file
    """
    
    AllMetaData_signal=get_JSON(signal_dir_path)
    if AllMetaData_signal == {}:
        print('no signaling -> skip')
        return False
    res=AllMetaData_signal

    AllMetaData_vol=get_JSON(vol_dir_path)
    if AllMetaData_vol == {}:
        print('no vol -> skip')
        return False
    res.update(AllMetaData_vol)

    AllMetaData_LM=get_JSON(LMcoord_dir_path)
    if not 'Coord_MetaData' in AllMetaData_LM:
        print('no Surface(Coord) -> skip')
        return False

    res.update(AllMetaData_LM)

    AllMetaData_proj=get_JSON(project_dir_path)

    try:
        if AllMetaData_proj['proj_MetaData']['proj version']!=proj_version:
            print('different proj version')
            return res
        if  AllMetaData_proj['proj_MetaData']['input Surface checksum']!=res['Coord_MetaData']['output Surface checksum']:
            print(AllMetaData_proj['proj_MetaData']['input Surface checksum'])
            print(res['Coord_MetaData']['output Surface checksum'])
            print('differnt surface')
            return res  
        if  AllMetaData_proj['proj_MetaData']['input vol checksum']!=res['vol_image_MetaData']['output vol checksum']:
            print('differnt vol')
            return res  
        if  AllMetaData_proj['proj_MetaData']['input '+ending+ ' checksum']!=res[ending+'_image_MetaData'][ending+ ' checksum']:
            print('differnt sig')
            return res  
        print('already done')
        return False    #skip

    except:
        return res  #corrupted -> do again
     
def make_projections():
    ending='Smoc'
    signal_images_path=os.path.join(structured_data_path,'images','RAW_images_and_splitted','raw_images_'+ending)
    signal_folder_list=os.listdir(signal_images_path)
    for signal_folder in signal_folder_list:
        print(signal_folder)

        LMcoord_dir_path=os.path.join(LMcoord_path,signal_folder[:-len('_'+ending)]+'_nuclei_LMcoord')
        vol_dir_path=os.path.join(vol_images_path,signal_folder[:-len('_'+ending)]+'_vol')
        signal_dir_path=os.path.join(signal_images_path,signal_folder)
        #PR_dir_path=os.path.join(proj_path,'point_regions',signal_folder[:-len('_'+ending)]+ "_PR") 
        project_dir_path=os.path.join(proj_path,ending,signal_folder+'_proj') 
        make_path(project_dir_path)
        
        PastMetaData=evalStatus_proj(signal_dir_path,vol_dir_path,LMcoord_dir_path,project_dir_path,ending)
        if not isinstance(PastMetaData,dict):
            continue

        MetaData_vol=PastMetaData['vol_image_MetaData']
        vol_file=MetaData_vol['vol image file name']
        vol_file_path=os.path.join(vol_dir_path,vol_file)
        signal_file_path=os.path.join(signal_dir_path,PastMetaData[ending+'_image_MetaData'][ending+' image file name'])
        scales=MetaData_vol['XYZ size in mum'].copy()
        if MetaData_vol['axes']=='ZYX':
            scales[0], scales[-1] = scales[-1], scales[0]
        MetaData_Surface=PastMetaData['Surface_MetaData']
        Surface_file=MetaData_Surface['Surface file']
        Surface_file_path=os.path.join(LMcoord_dir_path,Surface_file)
        #actual calculation
        print('compute proj')
        proj_mesh=project(signal_file_path,vol_file_path,Surface_file_path,scales,ending)
        if proj_mesh is None:
            print('error')
            continue

        proj_file_name=signal_folder+'_proj.vtk'
        proj_file=os.path.join(project_dir_path,proj_file_name)
        proj_mesh.save(proj_file)        
        
        MetaData_proj={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_proj['git hash']=sha
        MetaData_proj['git repo']='meshProjection'
        MetaData_proj['proj version']=proj_version
        MetaData_proj['proj file']=proj_file_name
        MetaData_proj['XYZ size in mum']=PastMetaData['Coord_MetaData']['XYZ size in mum']
        MetaData_proj['axes']=PastMetaData['Coord_MetaData']['axes']
        MetaData_proj['experimentalist']=PastMetaData['Coord_MetaData']['experimentalist']
        MetaData_proj['is control']=PastMetaData['Coord_MetaData']['is control']
        MetaData_proj['genotype']=PastMetaData['Coord_MetaData']['genotype']
        MetaData_proj['time in hpf']=PastMetaData['Coord_MetaData']['time in hpf']
        MetaData_proj['input Surface checksum']=PastMetaData['Coord_MetaData']['output Surface checksum']
        MetaData_proj['input vol checksum']=PastMetaData['vol_image_MetaData']['output vol checksum']
        MetaData_proj['input '+ending+ ' checksum']=PastMetaData[ending+'_image_MetaData'][ending+ ' checksum']
        check_proj=get_checksum(proj_file, algorithm="SHA1")
        MetaData_proj['output proj checksum']=check_proj
        writeJSON(project_dir_path,'proj_MetaData',MetaData_proj)


if __name__ == "__main__":
    make_projections() 
