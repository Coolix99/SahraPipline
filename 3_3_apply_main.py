import numpy as np
#np.random.seed(42)
from typing import List
import napari
import os
import git
from simple_file_checksum import get_checksum
import pyvista as pv
import random
from scipy.spatial import cKDTree

from cellpose import models
from cellpose.core import run_net
from cellpose.models import Cellpose

from config import *
from IO import *

def compute_curve_patch(model:Cellpose,img):
    masks, flows, _=model.cp.eval(img)

    return flows[1]

def calculate_flow(mesh:pv.PolyData,proj,model):
    bsize=224 #224?
    im_scale=1
    num_points = mesh.points.shape[0]

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
        #mesh.compute_normals(point_normals=True, cell_normals=True, auto_orient_normals=True, flip_normals=False)
        mesh.plot_normals(mag=5)
        return
        n=mesh.point_normals[random_index]
        e1=np.array((1,0,0))
        e1=e1-np.dot(e1,n)
        e1=e1/np.linalg.norm(e1)
        e2=np.cross(e1,n)

        vertices = mesh.points
        faces = mesh.faces.reshape((-1, 4))[:, 1:4]
        viewer = napari.Viewer()
        viewer.add_surface((vertices, faces, proj), name='Surface Mesh')
        vectors = np.stack([n])  # Assuming e1, e2, n are already defined as unit vectors
        origins = np.array([r0])  # Originating from r0
        vector_data = np.concatenate([origins[:, np.newaxis, :], origins[:, np.newaxis, :] + vectors[:, np.newaxis, :]], axis=1)
        viewer.add_vectors(vector_data, edge_width=2, edge_color=['cyan'], name='Vectors at r0', length=0.1)
        napari.run()
        return

        indices = np.arange(-bsize//2, bsize//2 )

        x = np.tile(indices, (bsize, 1))
        y = np.tile(indices[:, np.newaxis], (1, bsize))
        pos=x[:,:,np.newaxis]*e1[np.newaxis,np.newaxis,:]*im_scale+y[:,:,np.newaxis]*e2[np.newaxis,np.newaxis,:]*im_scale+r0[np.newaxis,np.newaxis,:]

        flattened_pos = pos.reshape(-1, 3)

        import trimesh
        vertices = mesh.points 
        faces = mesh.faces.reshape(-1, 4)[:, 1:]  
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(tri_mesh)
        ray_origins = pos.reshape(-1, 3)
        ray_directions_n = np.tile(n, (ray_origins.shape[0], 1))
        locations_n, index_ray_n, index_tri_n = ray_intersector.intersects_location(ray_origins, ray_directions_n, multiple_hits=False)
        ray_directions_neg_n = -ray_directions_n  
        locations_neg_n, index_ray_neg_n, index_tri_neg_n = ray_intersector.intersects_location(ray_origins, ray_directions_neg_n, multiple_hits=False)

        print(index_ray_n.shape)
        print(index_ray_neg_n.shape)
       
        distances_n = np.linalg.norm(locations_n - ray_origins[index_ray_n], axis=1)
        distances_neg_n = np.linalg.norm(locations_neg_n - ray_origins[index_ray_neg_n], axis=1)
        print(distances_n.shape)
        print(distances_neg_n.shape)

        arr_distances_n=np.full(ray_origins.shape[0], np.inf)
        arr_distances_neg_n=np.full(ray_origins.shape[0], np.inf)
        arr_distances_n[index_ray_n]=distances_n
        arr_distances_neg_n[index_ray_neg_n]=distances_neg_n
        
        arr_locations_neg_n = np.full_like(ray_origins, np.nan, dtype=np.float64)
        arr_locations_n = np.full_like(ray_origins, np.nan, dtype=np.float64)
        arr_locations_n[index_ray_n]=locations_n
        arr_locations_neg_n[index_ray_neg_n]=locations_neg_n
        
        # Initialize an array to store the minimum distances and corresponding locations
        min_distances = np.full(ray_origins.shape[0], np.inf)
        nearest_locations = np.full_like(ray_origins, np.nan, dtype=np.float64)
        #final_index_ray = np.full(ray_origins.shape[0], -1, dtype=int)  

        mask=arr_distances_n<arr_distances_neg_n
        nearest_locations[mask]=arr_locations_n[mask]
        nearest_locations[~mask]=arr_locations_neg_n[~mask]

        min_distances[mask]=arr_distances_n[mask]
        min_distances[~mask]=arr_distances_neg_n[~mask]


        valid_indices = ~np.isnan(nearest_locations).any(axis=1)
        locations = nearest_locations[valid_indices]
        index_ray = np.where(valid_indices)


        vertices = mesh.points
        faces = mesh.faces.reshape((-1, 4))[:, 1:4]
        viewer = napari.Viewer()
        viewer.add_surface((vertices, faces, proj), name='Surface Mesh')
        viewer.add_points(locations, size=3, face_color='blue', edge_color='blue', name='Intersection Points')
        viewer.add_points(flattened_pos, size=3, face_color='red', edge_color='red', name='Position Points')
        viewer.add_points([r0], size=10, face_color='green', edge_color='green', name='Random Point r0')
        vectors = np.stack([e1, e2, n])  # Assuming e1, e2, n are already defined as unit vectors
        origins = np.array([r0, r0, r0])  # Originating from r0
        vector_data = np.concatenate([origins[:, np.newaxis, :], origins[:, np.newaxis, :] + vectors[:, np.newaxis, :]], axis=1)
        viewer.add_vectors(vector_data, edge_width=2, edge_color=['cyan', 'magenta', 'yellow'], name='Vectors at r0', length=0.1)
        napari.run()
        
        return

        print(locations.shape)
        print(index_ray.shape)
        ind_ray = np.unique(index_ray, return_index=True)[1]

        tree = cKDTree(mesh.points)
        _, ind_mesh = tree.query(locations)
        

        image_patch=np.zeros((ray_origins.shape[0]))
        image_patch[ind_ray]=proj[ind_mesh]

        image_patch=image_patch.reshape(bsize, bsize)
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns, and optional figure size

        # Second subplot for image_patch
        img2 = axs[1].imshow(image_patch, cmap='gray', origin='lower')
        fig.colorbar(img2, ax=axs[1])  # Add a color bar to the subplot
        axs[1].set_title('Image Patch in Grayscale')

        # Show the plot
        plt.show()

        #print(index_tri)
        # # Handle the results to get the first intersection data per ray
        # 
        # first_intersections = locations[unique_indices]
        # first_ray_indices = index_ray[unique_indices]

        # # Extracting vertex colors (or any other point data you have, e.g., 'proj')
        # first_color_data = mesh.visual.vertex_colors[mesh.faces[index_tri[unique_indices]]][:,0,:]

        # # Map data back to the grid shape
        # intersected_data = np.full((x.size, 4), np.nan)  # Prepare an array full of NaNs
        # intersected_data[first_ray_indices] = first_color_data  # Place the color data where intersections happened
        # intersected_data = intersected_data.reshape(x.shape + (4,))  # Reshape to the original grid shape with color channels

        
        return


        tree = cKDTree(mesh.points)
        distances, indices = tree.query(flattened_pos)
        image_data = proj[indices]
        image_patch = image_data.reshape(bsize, bsize)
        distance_patch = distances.reshape(bsize, bsize)

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns, and optional figure size

        # First subplot for distance_patch
        img1 = axs[0].imshow(distance_patch, cmap='viridis', origin='lower')
        fig.colorbar(img1, ax=axs[0])  # Add a color bar to the subplot
        axs[0].set_title('Distance Patch Visualization')

        # Second subplot for image_patch
        img2 = axs[1].imshow(image_patch, cmap='gray', origin='lower')
        fig.colorbar(img2, ax=axs[1])  # Add a color bar to the subplot
        axs[1].set_title('Image Patch in Grayscale')

        # Show the plot
        plt.show()
        return

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

        dist_thr=2.5
        mask=distance_patch<dist_thr

        # Transforming 2D flow to 3D using e1 and e2
        flow_3d = flow[0, :, :, np.newaxis] * e1[np.newaxis, np.newaxis, :] + flow[1, :, :, np.newaxis] * e2[np.newaxis, np.newaxis, :]
        flow_3d = flow_3d.reshape(bsize * bsize, 3)

        # Apply mask
        valid_flows = flow_3d[mask.reshape(-1), :]
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
