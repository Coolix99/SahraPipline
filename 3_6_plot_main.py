import pyvista as pv
import numpy as np
from typing import List
import napari
import os
import matplotlib.pyplot as plt

from config import *
from IO import *

def plot_img_res(mesh,proj,labels,sig_img,scales):
    unique_integers = np.unique(labels)
    colors = plt.cm.get_cmap('nipy_spectral', len(unique_integers))
    np.random.seed(42//2) 
    indices = np.arange(len(unique_integers))
    np.random.shuffle(indices)
    color_mapping = {int_val: colors(idx / len(unique_integers)) for idx, int_val in zip(indices, unique_integers)}
    vertex_colors = np.array([color_mapping[val] for val in labels])
    viewer = napari.Viewer(ndisplay=3)
    vertices = mesh.points/scales
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]
    viewer.add_surface((vertices, faces),vertex_colors=vertex_colors, name='Seg res')
    viewer.add_surface((vertices, faces, proj), name='proj')
    viewer.add_image(sig_img, name='img')
    napari.run()

def plot_res(mesh,proj,labels):
    unique_integers = np.unique(labels)
    colors = plt.cm.get_cmap('nipy_spectral', len(unique_integers))
    np.random.seed(42//2) 
    indices = np.arange(len(unique_integers))
    np.random.shuffle(indices)
    color_mapping = {int_val: colors(idx / len(unique_integers)) for idx, int_val in zip(indices, unique_integers)}
    vertex_colors = np.array([color_mapping[val] for val in labels])
    viewer = napari.Viewer(ndisplay=3)
    vertices = mesh.points
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]
    viewer.add_surface((vertices, faces),vertex_colors=vertex_colors, name='Seg res')
    viewer.add_surface((vertices, faces, proj), name='proj')
    napari.run()

def plot():
    signal_dir_path=os.path.join(EpFlat_path,'signal_upperlayer')
    fin_list=[folder for folder in os.listdir(EpSeg_path) if os.path.isdir(os.path.join(EpSeg_path, folder))]
    for fin_folder in fin_list:
        print(fin_folder)
    
        fin_dir_path=os.path.join(EpSeg_path,fin_folder)
        
        PastMetaData=get_JSON(fin_dir_path)
        if not isinstance(PastMetaData,dict):
            continue
        if not 'Mesh_MetaData' in PastMetaData:
            continue


        MetaData_mesh=PastMetaData['Mesh_MetaData']
        scales=MetaData_mesh['scales'].copy()
        Mesh_file=MetaData_mesh['Mesh file']
        Mesh_file_path=os.path.join(fin_dir_path,Mesh_file)
        mesh=pv.read(Mesh_file_path)

        MetaData_proj=PastMetaData['project_MetaData']
        proj_file=MetaData_proj['proj file']
        proj_file_path=os.path.join(fin_dir_path,proj_file)
        proj=loadArr(proj_file_path)

        MetaData_apply=PastMetaData['seg_MetaData']
        seg_file=MetaData_apply['seg file']
        seg_file_path=os.path.join(fin_dir_path,seg_file)
        labels=loadArr(seg_file_path)

        signal_path=os.path.join(signal_dir_path,fin_folder+'.tif')
        sig_img=getImage(signal_path)

        # print(scales)
        # print(proj)
        # print(labels)
        # print(sig_img)

        #plot_img_res(mesh,proj,labels,sig_img,scales)
        plot_res(mesh,proj,labels)

if __name__ == "__main__":
    plot() 
