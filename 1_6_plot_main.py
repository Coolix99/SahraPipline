import numpy as np
from typing import List
import napari
import os
import git
from simple_file_checksum import get_checksum
import pyvista as pv

from config import *
from IO import *


def plot_surface(surf_file):
    mesh=pv.read(surf_file)
    print(mesh.point_data)
    mesh.plot(scalars='thickness')

def plot_surface_image_mask(surf_file,membrane_file, ED_file, mask_file,scales):
    mesh=pv.read(surf_file)
    print(mesh.point_data)

    membrane_img=getImage(membrane_file)
    ED_img=getImage(ED_file)
    mask_img=getImage(mask_file)

    thickness = mesh.point_data['thickness']
    vertices = mesh.points/np.array(scales)
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]

    viewer = napari.Viewer()
    
    # Add the mesh as a surface layer
    viewer.add_surface((vertices, faces, thickness), colormap='viridis', name='Mesh Thickness')
    # Add images to the viewer
    viewer.add_image(membrane_img, name='Membrane Image')
    viewer.add_image(ED_img, name='ED Image')
    viewer.add_image(mask_img, name='Mask Image')
    
    # Start Napari viewer
    napari.run()
    

def plot_all():
    FlatFin_folder_list=os.listdir(FlatFin_path)
    FlatFin_folder_list = [item for item in FlatFin_folder_list if os.path.isdir(os.path.join(FlatFin_path, item))]
    for FlatFin_folder in FlatFin_folder_list:
        print(FlatFin_folder)
        FlatFin_dir_path=os.path.join(FlatFin_path,FlatFin_folder)

        MetaData=get_JSON(FlatFin_dir_path)
        if not 'Surface_MetaData' in MetaData:
            print('no surface')
            continue
        if 'Thickness_MetaData' in MetaData:
            MetaData=MetaData['Thickness_MetaData']
        elif 'Coord_MetaData' in MetaData:
            MetaData=MetaData['Coord_MetaData']
        else:
            MetaData=MetaData['Surface_MetaData']


        Surface_file_name=MetaData['Surface file']

        #plot_surface(os.path.join(FlatFin_dir_path,Surface_file_name))

        data_name=FlatFin_folder[:-len('_FlatFin')]
        
        plot_surface_image_mask(os.path.join(FlatFin_dir_path,Surface_file_name),
                                os.path.join(membranes_path,data_name,data_name+'.tif'),
                                os.path.join(ED_marker_path,data_name,data_name+'.tif'),
                                os.path.join(finmasks_path,data_name,data_name+'.tif'),
                                MetaData['scales ZYX'])

if __name__ == "__main__":
    plot_all() 

