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

    mesh.plot()


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

        plot_surface(os.path.join(FlatFin_dir_path,Surface_file_name))

if __name__ == "__main__":
    plot_all() 

