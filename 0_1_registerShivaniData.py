import numpy as np
from typing import List,Tuple
#import napari
import os
import git
#import pandas as pd
from simple_file_checksum import get_checksum
import re
import czifile
import xml.etree.ElementTree as ET

from config import *
from IO import *


def extract_info(s: str) -> Tuple[int, bool]:
    # Extract the number before 'h'
    match = re.search(r'(\d+)h', s)
    if match:
        hours = int(match.group(1))
    else:
        raise ValueError("The string does not contain a valid 'h' pattern with a preceding number.")
    
    # Check if '_reg' is present in the string
    is_reg = '_reg' in s
    
    return hours, not is_reg

def read_czi_metadata(file_path: str):
    # Open the .czi file
    with czifile.CziFile(file_path) as czi:
        # Read  metadata
        metadata = czi.metadata()
        root = ET.fromstring(metadata)
        voxel_size = {}
        for dim in ['X', 'Y', 'Z']:
            distance_element = root.find(f".//Scaling/Items/Distance[@Id='{dim}']")
            if distance_element is not None:
                voxel_size[dim] = float(distance_element.find('Value').text) * 1e6  # Convert meters to micrometers

        image_data = np.squeeze(czi.asarray())

    return voxel_size, image_data

def register_raw():
    raw_folders_path=os.path.join(Input_Shivani_path,'raw_images')
    
    raw_folders_list= [item for item in os.listdir(raw_folders_path) if os.path.isdir(os.path.join(raw_folders_path, item))]
    for raw_folder in raw_folders_list:
        time, is_dev=extract_info(raw_folder)
        print(time, is_dev)
        img_folder_path=os.path.join(raw_folders_path,raw_folder)
        im_list = os.listdir(img_folder_path)
        for img_name in im_list:
            print(img_name)
            
            voxel_size_um, image_data=read_czi_metadata(os.path.join(img_folder_path,img_name))
            print(voxel_size_um)
            print(image_data.shape)
            
            # viewer = napari.Viewer()
            # for i in range(image_data.shape[0]):  # Iterate over the channels
            #     viewer.add_image(image_data[i], name=f'Channel {i+1}')

            # # Run the Napari event loop
            # napari.run()
            #remove ending
            img_name = os.path.splitext(img_name)[0]

            #Membrane
            membrane_folder_path=os.path.join(membranes_path,img_name)
            make_path(membrane_folder_path)
            membrane_im_path=os.path.join(membrane_folder_path,img_name+'.tif')
            save_array_as_tiff(image_data[0],membrane_im_path)

            MetaData_membrane={}
            repo = git.Repo(gitPath,search_parent_directories=True)
            sha = repo.head.object.hexsha
            MetaData_membrane['git hash']=sha
            MetaData_membrane['git repo']='Sahrapipline'
            MetaData_membrane['membrane file']=img_name+'.tif'
            MetaData_membrane['scales ZYX']=[voxel_size_um['Z'],voxel_size_um['Y'],voxel_size_um['X']]
            MetaData_membrane['condition']='Development' if is_dev else 'Regeneration'
            MetaData_membrane['time in hpf']=time
            MetaData_membrane['experimentalist']='Shivani'
            MetaData_membrane['genotype']='WT'
            check=get_checksum(membrane_im_path, algorithm="SHA1")
            MetaData_membrane['membrane checksum']=check
            writeJSON(membrane_folder_path,'MetaData_membrane',MetaData_membrane)

            #ED_marker
            ED_marker_folder_path=os.path.join(ED_marker_path,img_name)
            make_path(ED_marker_folder_path)
            ED_marker_im_path=os.path.join(ED_marker_folder_path,img_name+'.tif')
            save_array_as_tiff(image_data[0],ED_marker_im_path)

            MetaData_ED_marker={}
            repo = git.Repo(gitPath,search_parent_directories=True)
            sha = repo.head.object.hexsha
            MetaData_ED_marker['git hash']=sha
            MetaData_ED_marker['git repo']='Sahrapipline'
            MetaData_ED_marker['ED_marker file']=img_name+'.tif'
            MetaData_ED_marker['scales ZYX']=[voxel_size_um['Z'],voxel_size_um['Y'],voxel_size_um['X']]
            MetaData_ED_marker['condition']='Development' if is_dev else 'Regeneration'
            MetaData_ED_marker['time in hpf']=time
            MetaData_ED_marker['experimentalist']='Shivani'
            MetaData_ED_marker['genotype']='WT'
            check=get_checksum(ED_marker_im_path, algorithm="SHA1")
            MetaData_ED_marker['ED_marker checksum']=check
            writeJSON(ED_marker_folder_path,'MetaData_ED_marker',MetaData_ED_marker)

       
       

if __name__ == "__main__":
    register_raw()
