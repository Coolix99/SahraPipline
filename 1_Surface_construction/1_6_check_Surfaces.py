import numpy as np
from typing import List
import napari
import os
import git
from simple_file_checksum import get_checksum
import pyvista as pv
import shutil

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

def plot_surface_delete(Surface_file_name,data_name):
    FlatFin_dir_path=os.path.join(FlatFin_path,data_name+'_FlatFin')
    surf_file=os.path.join(FlatFin_dir_path,Surface_file_name)
    mesh = pv.read(surf_file)
    print(mesh.point_data)

    # Create a plotter
    plotter = pv.Plotter()

    # Add the mesh to the plotter
    plotter.add_mesh(mesh, scalars='thickness')

    # Define a callback function for key press events
    def key_callback(key):
        print(key)
        if key == 'd' or key == 'r':
            print(f"Deleting {surf_file}")
            if os.path.exists(FlatFin_dir_path):
                try:
                    shutil.rmtree(FlatFin_dir_path)
                except Exception as e:
                    print(f"Error: {e}")
            if key == 'r':
                print(f"Deleting also {data_name}")
                # shutil.rmtree(os.path.join(membranes_path,data_name))
                # shutil.rmtree(os.path.join(ED_marker_path,data_name))
                shutil.rmtree(os.path.join(finmasks_path,data_name))
            plotter.close()
            return False

    # Add the key press callback to the plotter
    plotter.add_key_event('d', lambda: key_callback('d'))  # Bind 'd' to delete
    plotter.add_key_event('r', lambda: key_callback('r'))  # Bind 'r' to delete
    plotter.add_key_event('q', lambda: key_callback('q'))  # Bind 'q' to close without deleting

    # Show the plot and wait for key press
    try:
        plotter.show()
    except Exception as e:
        print(f"Error: {e}")
    return True


def plot_all(skip_shown=True):
    FlatFin_folder_list=os.listdir(FlatFin_path)
    FlatFin_folder_list = [item for item in FlatFin_folder_list if os.path.isdir(os.path.join(FlatFin_path, item))]
    for FlatFin_folder in FlatFin_folder_list:
        print(FlatFin_folder)
        FlatFin_dir_path=os.path.join(FlatFin_path,FlatFin_folder)
        
        status_file = os.path.join(FlatFin_dir_path, 'shown.txt')
        if skip_shown:    
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = f.read().strip()
                    if status == 'shown':
                        print(f'Skipping {FlatFin_folder}, already shown.')
                        continue

        MetaData=get_JSON(FlatFin_dir_path)
        if not 'Surface_MetaData' in MetaData:
            print('no surface')
            continue
        if 'Thickness_MetaData' in MetaData:
            MetaData=MetaData['Thickness_MetaData']
        else:
            print('no Thickness')
            continue

        # elif 'Coord_MetaData' in MetaData:
        #     MetaData=MetaData['Coord_MetaData']
        # else:
        #     MetaData=MetaData['Surface_MetaData']


        Surface_file_name=MetaData['Surface file']
        #plot_surface(os.path.join(FlatFin_dir_path,Surface_file_name))

        data_name=FlatFin_folder[:-len('_FlatFin')]
        with open(status_file, 'w') as f:
            f.write('shown')
        
        plot_surface_delete(Surface_file_name,data_name)
        
        continue
        
        plot_surface_image_mask(os.path.join(FlatFin_dir_path,Surface_file_name),
                                os.path.join(membranes_path,data_name,data_name+'.tif'),
                                os.path.join(ED_marker_path,data_name,data_name+'.tif'),
                                os.path.join(finmasks_path,data_name,data_name+'.tif'),
                                MetaData['scales ZYX'])

if __name__ == "__main__":
    plot_all() 

