import numpy as np
from typing import List,Tuple
import napari
import os
import git
#import pandas as pd
from simple_file_checksum import get_checksum
import re

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
    is_reg = '_reg' in s
    
    return hours, not is_reg

def register_EDcells(skip_existing=True):
    EDcells_folders_path=os.path.join(Input_Shivani_path,'segmented_ED_cells')
    
    Folder_folders_list= [item for item in os.listdir(EDcells_folders_path) if os.path.isdir(os.path.join(EDcells_folders_path, item))]
    for Folder in Folder_folders_list:
        print(Folder)
        
        time, is_dev=extract_info(Folder)
        print(time, is_dev)
        Folder_path=os.path.join(EDcells_folders_path,Folder)
        im_list = os.listdir(Folder_path)
        tif_files = [img_name for img_name in im_list if img_name.lower().endswith('.tif')]
        for img_name in tif_files:
            print(img_name)
            
            EDcells_folder_path=os.path.join(ED_cells_path,os.path.splitext(img_name)[0])
            make_path(EDcells_folder_path)
            if (not get_JSON(EDcells_folder_path)=={}) and skip_existing:
                print('skip')
                continue

            im=getImage(os.path.join(Folder_path,img_name))
            
            EDcells_im_path=os.path.join(EDcells_folder_path,img_name)
            save_array_as_tiff(im,EDcells_im_path)

            MetaData_EDcells={}
            repo = git.Repo(gitPath,search_parent_directories=True)
            sha = repo.head.object.hexsha
            MetaData_EDcells['git hash']=sha
            MetaData_EDcells['git repo']='Sahrapipline'
            MetaData_EDcells['EDcells file']=img_name
            MetaData_EDcells['condition']='Development' if is_dev else 'Regeneration'
            MetaData_EDcells['time in hpf']=time
            MetaData_EDcells['genotype']='WT'
            check=get_checksum(EDcells_im_path, algorithm="SHA1")
            MetaData_EDcells['EDcells checksum']=check
            writeJSON(EDcells_folder_path,'MetaData_EDcells',MetaData_EDcells)       
       

if __name__ == "__main__":
    register_EDcells(False)
