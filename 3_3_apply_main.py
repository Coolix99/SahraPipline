import numpy as np
from typing import List
import napari
import os
import git
from simple_file_checksum import get_checksum
import pyvista as pv
import random
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from scipy.interpolate import griddata

from cellpose import models
from cellpose.core import run_net
from cellpose.models import Cellpose

from config import *
from IO import *

def compute_curve_patch(model:Cellpose,img):
    masks, flows, _=model.cp.eval(img)

    return flows[1]

def getPatchImg(e1,e2,n,r0,bsize,im_scale,points,proj,random_index):
    relpos=points-r0
    im_scale=1
    rel_1 = np.dot(relpos, e1) * im_scale + bsize // 2
    rel_2 = np.dot(relpos, e2) * im_scale + bsize // 2
    rel_n = np.abs(np.dot(relpos, n))
    mask = (rel_1 >= -10) & (rel_1 <= bsize + 10) & (rel_2 >= -10) & (rel_2 <= bsize + 10) & (rel_n < 30)
    filtered_points = np.vstack((rel_1[mask], rel_2[mask])).T
    filtered_proj = proj[mask]
    filtered_distance = rel_n[mask]

    # Create empty patches
    image_patch = np.zeros((bsize, bsize))
    distance_patch = np.zeros((bsize, bsize))

    # Generate grid for interpolation
    grid_x, grid_y = np.mgrid[0:bsize, 0:bsize]

    # Check if we have enough points to form a triangulation
    if len(filtered_points) >= 3:
        # Interpolate the projection data and distances onto the grid
        image_patch = griddata(filtered_points, filtered_proj, (grid_x, grid_y), method='linear', fill_value=0)
        distance_patch = griddata(filtered_points, filtered_distance, (grid_x, grid_y), method='linear', fill_value=np.inf)

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 6))
        # plt.triplot(filtered_points[:, 0], filtered_points[:, 1], tri.simplices, 'ko-')
        # #plt.tripcolor(filtered_points[:, 0], filtered_points[:, 1], tri.simplices, facecolors=filtered_proj, edgecolors='k', cmap='viridis')
        # #plt.colorbar(label='Projection Values')
        # plt.title('Delaunay Triangulation with Projection Values')
        # plt.xlabel('Relative X')
        # plt.ylabel('Relative Y')
        # plt.grid(True)
        # plt.show()


    else:
        # If not enough points, fill with default values (can adjust as needed)
        image_patch.fill(0)
        distance_patch.fill(np.inf)

    return image_patch, distance_patch

def getPatchImg_old(e1, e2, n, r0, bsize, im_scale, mesh, proj):
    indices = np.arange(-bsize//2, bsize//2)
    x = np.tile(indices, (bsize, 1))
    y = np.tile(indices[:, np.newaxis], (1, bsize))
    pos = x[:, :, np.newaxis] * e1[np.newaxis, np.newaxis, :] * im_scale + \
          y[:, :, np.newaxis] * e2[np.newaxis, np.newaxis, :] * im_scale + \
          r0[np.newaxis, np.newaxis, :]
    
    # Initialize patches
    image_patch = np.zeros((bsize, bsize))
    distance_patch = np.full((bsize, bsize), np.inf)  # Start with inf distances
    
    # Ray tracing for each point in the grid
    for i in range(bsize):
        for j in range(bsize):
            # Define ray start and end points
            origin = pos[i, j] - n * 10
            end_point = origin + n * 20  
            
            # Perform ray tracing using PyVista
            first_intersection, intersection_cells = mesh.ray_trace(origin, end_point, first_point=True)
            if first_intersection.size > 0:  # Check if there is one intersection
                # Calculate the distance from origin to the intersection point
                distance_patch[i, j] = np.linalg.norm(first_intersection - origin)
                
                # Assuming 'proj' is accessible by point indices and is mapped correctly in 'mesh'
                # Find the closest point in the mesh to the intersection point
                idx = mesh.find_closest_point(first_intersection)
                image_patch[i, j] = proj[idx]
            else:
                # No intersection found, set the image patch value to 0
                image_patch[i, j] = 0

    return image_patch, distance_patch


def calculate_flow(mesh:pv.PolyData,proj,model):
    bsize=224
    im_scale=1
    num_points = mesh.points.shape[0]
    print(num_points)

    res_flow=np.zeros((3,num_points))
    add_flows=np.zeros_like(res_flow)
    n_flows=np.zeros_like(res_flow,dtype=int)

    while True:
        open_ind=np.where(n_flows[0,:] <1)[0]
        print(open_ind.shape[0])
        if open_ind.shape[0]<1:
            break

        random_index = np.random.choice(open_ind)
        
        r0=mesh.points[random_index]
        n=mesh.point_normals[random_index]
        e1=np.array((1,0,0))
        e1=e1-np.dot(e1,n)
        e1=e1/np.linalg.norm(e1)
        e2=np.cross(e1,n)

        return

        image_patch,distance_patch=getPatchImg(e1,e2,n,r0,bsize,im_scale,mesh.points,proj,random_index)
        sigma = 2.0  # Define the amount of blurring
        from scipy.ndimage import gaussian_filter
        distance_patch = gaussian_filter(distance_patch, sigma=sigma)
        bad_pxls=distance_patch>25.0
        image_patch[bad_pxls]=0

        import matplotlib.pyplot as plt
        plt.imshow(distance_patch, cmap='viridis', origin='lower')  # You can change the colormap as needed
        plt.colorbar()  # Optionally add a color bar to show the mapping of color to data values
        plt.title('Dist from Mesh Point Data')
        plt.show()
        plt.imshow(image_patch, cmap='gray', origin='lower')  # You can change the colormap as needed
        plt.colorbar()  # Optionally add a color bar to show the mapping of color to data values
        plt.title('Image from Mesh Point Data')
        plt.show()
        
        flow=compute_curve_patch(model,image_patch)

        import matplotlib.pyplot as plt
        step = 2  
        x, y = np.meshgrid(np.arange(0, bsize, step), np.arange(0, bsize, step))
        fig, ax = plt.subplots()
        im = ax.imshow(image_patch, cmap='gray', origin='lower')
        scale_factor = 2  
        qv = ax.quiver(x, y, flow[0, ::step, ::step] * scale_factor, flow[1, ::step, ::step] * scale_factor, color='r')
        fig.colorbar(im, ax=ax, orientation='vertical')
        plt.title('Image with Vector Field Overlay')
        plt.show()

        dist_thr=10
        mask=distance_patch<dist_thr

        # Transforming 2D flow to 3D using e1 and e2
        flow_3d = flow[0, :, :, np.newaxis] * e1[np.newaxis, np.newaxis, :] + flow[1, :, :, np.newaxis] * e2[np.newaxis, np.newaxis, :]
        flow_3d = flow_3d.reshape(bsize * bsize, 3)

        # Apply mask
        valid_flows = flow_3d[mask.reshape(-1), :]
        indices = np.arange(-bsize//2, bsize//2 )
        x = np.tile(indices, (bsize, 1))
        y = np.tile(indices[:, np.newaxis], (1, bsize))
        pos=x[:,:,np.newaxis]*e1[np.newaxis,np.newaxis,:]*im_scale+y[:,:,np.newaxis]*e2[np.newaxis,np.newaxis,:]*im_scale+r0[np.newaxis,np.newaxis,:]
        flattened_pos = pos.reshape(-1, 3)
        valid_positions = flattened_pos[mask.reshape(-1), :]

        # Update flows at corresponding 3D positions using cKDTree
        tree = cKDTree(mesh.points)
        _, indices = tree.query(valid_positions)

        # Accumulate flows using advanced indexing
        np.add.at(add_flows, (slice(None), indices), valid_flows.T)
        np.add.at(n_flows, (slice(None), indices), 1)

        # Avoid division by zero and normalize flows
        valid_mask = n_flows > 0
        res_flow[valid_mask] = add_flows[valid_mask] / n_flows[valid_mask]

        valid_indices = n_flows[0,:] > 0
        magnitudes = np.linalg.norm(res_flow[:, valid_indices], axis=0)
        magnitudes[magnitudes == 0] = 1
        res_flow[:, valid_indices] /= magnitudes

        print(res_flow)
        print(res_flow[:, valid_indices])
        print(valid_indices.shape)

        vertices = mesh.points
        faces = mesh.faces.reshape((-1, 4))[:, 1:4]
        viewer = napari.Viewer()
        viewer.add_surface((vertices, faces, proj), name='Surface Mesh')
        n = 1
        sampled_points = vertices[::n]  # Reduce density for clarity
        sampled_flows = res_flow[:, ::n].T  # Transpose to align with napari's expected input
        vector_data = np.zeros((sampled_flows.shape[0], 2, 3))
        vector_data[:, 0, :] = sampled_points  # Start points
        vector_data[:, 1, :] =  sampled_flows *1  # End points, scale to adjust length

        # Adding the vector layer
        viewer.add_vectors(vector_data, edge_width=0.1, edge_color='cyan',
                        name='Flow Vectors')

        # Start napari
        napari.run()


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
