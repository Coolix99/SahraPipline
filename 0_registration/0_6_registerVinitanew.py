import numpy as np
from typing import List,Tuple
import napari
import os
import git
#import pandas as pd
from simple_file_checksum import get_checksum
import re
from scipy.ndimage import gaussian_filter
from imaris_ims_file_reader.ims import ims
import pyclesperanto_prototype as cle
from skimage.morphology import remove_small_holes
from scipy.ndimage import gaussian_filter
from skimage import measure


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    if 'reg' in s and 'dev' in s:
        raise ValueError("Both 'reg' and 'dev' are present in the string.")
    elif 'reg' in s:
        return hours, False
    elif 'dev' in s:
        return hours, True
    else:
        print(s)
        raise ValueError("Neither 'reg' nor 'dev' are present in the string.")

def load_ims(file):
    a = ims(file,ResolutionLevelLock=0)
    #print(a.shape)
    #print(a[0].shape)
    #print(a.ResolutionLevelLock)
    #print(a.ResolutionLevels)
    #print(a.TimePoints)
    #print(a.Channels)
    #print(a.shape)
    #print(a.chunks)
    #print(a.dtype)
    #print(a.ndim)
    #print(a.resolution)
    return a[0], a.resolution # type: ignore
   
def extract_hpf(string):
    """
    Extracts the integer before 'hpf' in the given string.
    
    Args:
        string (str): The input string containing the pattern ....{int}hpf.
        
    Returns:
        int: The integer value before 'hpf', or None if not found.
    """
    match = re.search(r'(\d+)hpf', string)
    if match:
        return int(match.group(1))  # Return the integer part
    return None  # Return None if pattern not found

def largest_connected_component(mask):
    # Label connected components
    labeled_mask, num_labels = measure.label(mask, background=0, return_num=True) # type: ignore
    
    # Count pixels in each connected component
    component_sizes = np.bincount(labeled_mask.ravel())
    
    # Get the index of the largest connected component
    largest_component_index = np.argmax(component_sizes[1:]) + 1
    
    # Create a mask with only the largest connected component
    largest_component_mask = (labeled_mask == largest_component_index)
    
    return largest_component_mask.astype(np.bool_)


def getVolthr(im,scales):
    mask=im>1
    disk_size_opening = int(18/scales[1])
    padded_mask = np.pad(mask, ((0, 0), (disk_size_opening, disk_size_opening), (disk_size_opening, disk_size_opening)), mode='constant', constant_values=0)
    print(padded_mask.shape)

    closed_mask = cle.closing_sphere(padded_mask,radius_x=disk_size_opening,radius_y=disk_size_opening,radius_z=0)

    closed_mask = closed_mask[:, disk_size_opening:closed_mask.shape[1]-disk_size_opening, disk_size_opening:closed_mask.shape[2]-disk_size_opening] # type: ignore
    processed_mask = np.zeros_like(mask,dtype=bool)
    for z in range(mask.shape[0]):
        opened_slice=np.array(closed_mask[z],dtype=bool)
        filled_slice = remove_small_holes(opened_slice, area_threshold=1000/scales[1]/scales[2]) 
        processed_mask[z] = filled_slice
    original_shape = im.shape
    return largest_connected_component(processed_mask[:original_shape[0], :original_shape[1], :original_shape[2]])

def register_finmask_special(skip_existing=True):
    Vinita_folders_path='/media/grp07_max/share_vinita/Smoc1_smoc2_dev'

    time_folders_list= [item for item in os.listdir(Vinita_folders_path) if os.path.isdir(os.path.join(Vinita_folders_path, item))]
 

    for time_folder in time_folders_list:
        time_hpf=extract_hpf(time_folder)
        print(time_folder,time_hpf)
        
        time_folder_path=os.path.join(Vinita_folders_path,time_folder)

        ims_files = [file for file in os.listdir(time_folder_path) if file.endswith('.ims')]
        for ims_file in ims_files:
            print(ims_file)
            
            finmasks_folder_path=os.path.join(finmasks_path,'vinita_smoc12_dev_'+time_folder+"_"+os.path.splitext(ims_file)[0])
            print(finmasks_folder_path)
            if os.path.exists(finmasks_folder_path) and skip_existing:
                print('skip')
                continue
            make_path(finmasks_folder_path)
            
            nucmem_folder_path=os.path.join(nucmem_path,'vinita_smoc12_dev_'+time_folder+"_"+os.path.splitext(ims_file)[0])
            make_path(nucmem_folder_path)
            bre_folder_path=os.path.join(bre_path,'vinita_smoc12_dev_'+time_folder+"_"+os.path.splitext(ims_file)[0])
            make_path(bre_folder_path)
            
            ims_file_path=os.path.join(time_folder_path,ims_file)
            im,voxel_size_um=load_ims(ims_file_path)
            print(im.shape) # type: ignore
            print(voxel_size_um)
            
            nuclei_and_membrane=im[-1,:,:,:] # type: ignore
            bre=im[0,:,:,:] # type: ignore
            
            mask=getVolthr(nuclei_and_membrane,voxel_size_um)
        
            # Minimal napari viewer for BRE, nuclei/membrane, and fin mask
            # viewer = napari.Viewer()

            # viewer.add_image(
            #     nuclei_and_membrane,
            #     name="Nuclei + membrane",
            #     scale=voxel_size_um,
            #     colormap="gray",
            # )

            # viewer.add_image(
            #     bre,
            #     name="BRE",
            #     scale=voxel_size_um,
            #     colormap="magenta",
            #     blending="additive",
            # )

            # viewer.add_labels(
            #     mask.astype(np.uint8),
            #     name="Fin mask",
            #     scale=voxel_size_um,
            #     opacity=0.4,
            # )

            # napari.run()
            # raise

            img_name='vinita_smoc12_dev_'+time_folder+"_"+os.path.splitext(ims_file)[0]+'.tif'
            finmasks_im_path=os.path.join(finmasks_folder_path,img_name)
            save_array_as_tiff(mask,finmasks_im_path)

            MetaData_finmasks={}
            repo = git.Repo(gitPath,search_parent_directories=True)
            sha = repo.head.object.hexsha
            MetaData_finmasks['git hash']=sha
            MetaData_finmasks['git repo']='Sahrapipline'
            MetaData_finmasks['finmasks file']=img_name
            MetaData_finmasks['condition']='Smoc12_Dev'
            MetaData_finmasks['scales ZYX']=[voxel_size_um[0],voxel_size_um[1],voxel_size_um[2]]
            MetaData_finmasks['time in hpf']=time_hpf
            MetaData_finmasks['experimentalist']='Vinita'
            MetaData_finmasks['genotype']='Smoc12'
            check=get_checksum(finmasks_im_path, algorithm="SHA1")
            MetaData_finmasks['finmasks checksum']=check
            writeJSON(finmasks_folder_path,'MetaData_finmasks',MetaData_finmasks)

            # #nuc
            # nucmem_im_path=os.path.join(nucmem_folder_path,img_name)
            # save_array_as_tiff(nuclei_and_membrane,nucmem_im_path)

            # MetaData_nucmem={}
            # repo = git.Repo(gitPath,search_parent_directories=True)
            # sha = repo.head.object.hexsha
            # MetaData_nucmem['git hash']=sha
            # MetaData_nucmem['git repo']='Sahrapipline'
            # MetaData_nucmem['nucmem file']=img_name
            # MetaData_nucmem['condition']='Smoc12_Dev'
            # MetaData_nucmem['scales ZYX']=[voxel_size_um[0],voxel_size_um[1],voxel_size_um[2]]
            # MetaData_nucmem['time in hpf']=time_hpf
            # MetaData_nucmem['genotype']='Vinita'
            # MetaData_nucmem['genotype']='Smoc12'
            # check=get_checksum(nucmem_im_path, algorithm="SHA1")
            # MetaData_nucmem['nucmem checksum']=check
            # writeJSON(nucmem_folder_path,'MetaData_nucmem',MetaData_nucmem)

            #bre
            bre_im_path=os.path.join(bre_folder_path,img_name)
            save_array_as_tiff(bre,bre_im_path)

            MetaData_bre={}
            repo = git.Repo(gitPath,search_parent_directories=True)
            sha = repo.head.object.hexsha
            MetaData_bre['git hash']=sha
            MetaData_bre['git repo']='Sahrapipline'
            MetaData_bre['bre file']=img_name
            MetaData_bre['condition']='Smoc12_Dev'
            MetaData_bre['scales ZYX']=[voxel_size_um[0],voxel_size_um[1],voxel_size_um[2]]
            MetaData_bre['time in hpf']=time_hpf
            MetaData_bre['genotype']='Vinita'
            MetaData_bre['genotype']='Smoc12'
            check=get_checksum(bre_im_path, algorithm="SHA1")
            MetaData_bre['bre checksum']=check
            writeJSON(bre_folder_path,'MetaData_bre',MetaData_bre)
        

      

if __name__ == "__main__":
    register_finmask_special()
