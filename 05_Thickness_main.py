import numpy as np
from typing import List
import napari
import os
import git
from simple_file_checksum import get_checksum
import pyvista as pv

from config import *
from IO import *


def getIntersections(vol_img,r0,normal,max_value=100,precision=1):
    a = 0
    b = max_value

    coord=np.clip(np.array((int(r0[0]+0.5+normal[0]*a),int(r0[1]+0.5+normal[1]*a),int(r0[2]+0.5+normal[2]*a)),dtype=int),[0,0,0],[vol_img.shape[0]-1,vol_img.shape[1]-1,vol_img.shape[2]-1])
    f_a=vol_img[coord[0],coord[1],coord[2]]
    if not f_a:
        return None

    while abs(b - a) > precision:
        midpoint = (a + b) / 2.0
        coord=np.clip(np.array((int(r0[0]+0.5+normal[0]*a),int(r0[1]+0.5+normal[1]*a),int(r0[2]+0.5+normal[2]*a)),dtype=int),[0,0,0],[vol_img.shape[0]-1,vol_img.shape[1]-1,vol_img.shape[2]-1])
        f_a=vol_img[coord[0],coord[1],coord[2]]
        coord=np.clip(np.array((int(r0[0]+0.5+normal[0]*midpoint),int(r0[1]+0.5+normal[1]*midpoint),int(r0[2]+0.5+normal[2]*midpoint)),dtype=int),[0,0,0],[vol_img.shape[0]-1,vol_img.shape[1]-1,vol_img.shape[2]-1])
        f_mid=vol_img[coord[0],coord[1],coord[2]]
        if f_mid == f_a:
            a = midpoint
        else:
            b = midpoint
    best_upper = (a + b) / 2.0

    a = 0
    b = -max_value
    while abs(b - a) > precision:
        midpoint = (a + b) / 2.0
        coord=np.clip(np.array((int(r0[0]+0.5+normal[0]*a),int(r0[1]+0.5+normal[1]*a),int(r0[2]+0.5+normal[2]*a)),dtype=int),[0,0,0],[vol_img.shape[0]-1,vol_img.shape[1]-1,vol_img.shape[2]-1])
        f_a=vol_img[coord[0],coord[1],coord[2]]
        coord=np.clip(np.array((int(r0[0]+0.5+normal[0]*midpoint),int(r0[1]+0.5+normal[1]*midpoint),int(r0[2]+0.5+normal[2]*midpoint)),dtype=int),[0,0,0],[vol_img.shape[0]-1,vol_img.shape[1]-1,vol_img.shape[2]-1])
        f_mid=vol_img[coord[0],coord[1],coord[2]]
        if f_mid == f_a:
            a = midpoint
        else:
            b = midpoint
    best_lower = (a + b) / 2.0

    return r0+normal*best_upper, r0+normal*best_lower

def calculate_Thickness(im_file,surf_file,scales):
    vol_img=getImage(im_file)
    mesh=pv.read(surf_file)
    import skimage as ski
    vol_img=ski.filters.gaussian(vol_img, sigma=(2,5,5),truncate=3)>0.5
    points_px=mesh.point_data['Coord px']
    normals_px=mesh.point_normals/scales

   
    upper_dist=np.zeros(points_px.shape[0],dtype=float)
    lower_dist=np.zeros(points_px.shape[0],dtype=float)
    for i in range(points_px.shape[0]):
        res=getIntersections(vol_img,points_px[i,:],normals_px[i,:])
        if res is None:
            upper_dist[i]=0
            lower_dist[i]=0
            continue
        upper_point, lower_point=res
        #print(upper_point,lower_point)
        upper_dist[i]=np.linalg.norm((upper_point-points_px[i,:])*scales)
        lower_dist[i]=np.linalg.norm((lower_point-points_px[i,:])*scales)
        
    mesh.point_data['upper_dist']=upper_dist
    mesh.point_data['lower_dist']=lower_dist
    mesh.point_data['thickness']=upper_dist+lower_dist

    # viewer = napari.Viewer(ndisplay=3)
    # faces = mesh.faces.reshape(-1, 4)[:, 1:]
    # surface = (points_px, faces)
    # viewer.add_surface(surface)
    # viewer.add_labels(vol_img)
    # #viewer.add_labels(blur)
    # viewer.add_points(points_plot)
    # napari.run()

    # p = pv.Plotter()
    # p.add_mesh(mesh,scalars="thickness", color="grey", ambient=0.6, opacity=0.5, show_edges=False)
    # p.show()
    # p = pv.Plotter()
    # p.add_mesh(mesh,scalars="lower_dist", color="grey", ambient=0.6, opacity=0.5, show_edges=False)
    # p.show()
    # p = pv.Plotter()
    # p.add_mesh(mesh,scalars="diff", color="grey", ambient=0.6, opacity=0.5, show_edges=False)
    # p.show()

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.hist(upper_dist-lower_dist, bins=30, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Histogram of Values Distribution')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()
    return mesh

def evalStatus_Thickness(FlatFin_dir_path):
    MetaData=get_JSON(FlatFin_dir_path)
    if not 'Orient_MetaData' in MetaData:
        print('no Orient_MetaData')
        return False

    if not 'CenterLine_MetaData' in MetaData:
        print('no CenterLine_MetaData')
        return False
    
    if not 'Surface_MetaData' in MetaData:
        print('no Surface_MetaData')
        return False
    
    if not 'Coord_MetaData' in MetaData:
        print('no Coord_MetaData')
        return False
    
    if not 'Thickness_MetaData' in MetaData:
        return MetaData

    if not MetaData['Thickness_MetaData']['Thickness version']==Thickness_version:
        return MetaData  

    if not MetaData['Thickness_MetaData']['input Surface checksum']==MetaData['Coord_MetaData']['output Surface checksum']:
        return MetaData

    return False

def make_Thickness():
    FlatFin_folder_list=os.listdir(FlatFin_path)
    FlatFin_folder_list = [item for item in FlatFin_folder_list if os.path.isdir(os.path.join(FlatFin_path, item))]
    for FlatFin_folder in FlatFin_folder_list:
        print(FlatFin_folder)
        FlatFin_dir_path=os.path.join(FlatFin_path,FlatFin_folder)
 
        PastMetaData=evalStatus_Thickness(FlatFin_dir_path)
        if not isinstance(PastMetaData,dict):
            continue
        data_name=FlatFin_folder[:-len('_FlatFin')]

        MetaData_Coord=PastMetaData['Coord_MetaData']
        Surface_file_name=MetaData_Coord['Surface file']

        scales=PastMetaData['CenterLine_MetaData']['scales'].copy()

        #actual calculation
        mesh = calculate_Thickness(os.path.join(vol_path,str(PastMetaData['Orient_MetaData']['time in hpf'])+'hpf',data_name+'.tif'),os.path.join(FlatFin_dir_path,Surface_file_name),scales)
        
        Surface_file=os.path.join(FlatFin_dir_path,Surface_file_name)
        mesh.save(Surface_file)

        MetaData_Thickness={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_Thickness['git hash']=sha
        MetaData_Thickness['git repo']='Sahrapipline'
        MetaData_Thickness['Thickness version']=Thickness_version
        MetaData_Thickness['Surface file']=Surface_file_name
        MetaData_Thickness['scales']=PastMetaData['Orient_MetaData']['scales']
        MetaData_Thickness['experimentalist']=PastMetaData['Orient_MetaData']['experimentalist']
        MetaData_Thickness['genotype']=PastMetaData['Surface_MetaData']['genotype']
        MetaData_Thickness['condition']=PastMetaData['Orient_MetaData']['condition']
        MetaData_Thickness['time in hpf']=PastMetaData['Surface_MetaData']['time in hpf']
        MetaData_Thickness['input Surface checksum']=PastMetaData['Coord_MetaData']['output Surface checksum']
        check_Surface=get_checksum(Surface_file, algorithm="SHA1")
        MetaData_Thickness['output Surface checksum']=check_Surface
        writeJSON(FlatFin_dir_path,'Thickness_MetaData',MetaData_Thickness)


if __name__ == "__main__":
    make_Thickness() 

