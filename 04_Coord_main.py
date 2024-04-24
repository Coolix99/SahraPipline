import numpy as np
np.random.seed(42)

from typing import List
import napari
import os
import git
import pandas as pd
from simple_file_checksum import get_checksum
from bisect import bisect_right
import pyvista as pv
import pymeshfix as mf
from scipy.spatial import cKDTree
import potpourri3d as pp3d
import numpy as np

from config import *
from IO import *




def calculate_positions_3d(x_positions, y_positions, z_positions,scales):
    x_diff = np.diff(x_positions)*scales[0]
    y_diff = np.diff(y_positions)*scales[1]
    z_diff = np.diff(z_positions)*scales[2]
    segment_distances = np.sqrt(x_diff*x_diff + y_diff*y_diff+ z_diff*z_diff)
    cumulative_distances = np.concatenate(([0], np.cumsum(segment_distances)))
    return cumulative_distances

def closest_point_on_segment(A, B, P):
    AB = B - A
    AP = P - A
    dot_product = np.sum(AP * AB, axis=1)
    denominator = np.sum(AB * AB, axis=1)
    factor = dot_product / denominator
    # Check if the factor is outside the [0, 1] range
    outside_segment = (factor < 0) | (factor > 1)
    # Set closest point to NaN if outside the segment
    closest_point = np.where(outside_segment[:, np.newaxis], np.inf, A + factor[:, np.newaxis] * AB)
    # Calculate distance between P and closest point X
    distance = np.linalg.norm(closest_point - P, axis=1)
    
    return closest_point, distance, factor

def closest_points_on_path(path,P):
    # Calculate the closest points and distances for each segment
    A_values = path[:-1]  # Starting points of segments
    B_values = path[1:]   # Ending points of segments
    closest_points, distances,_ = closest_point_on_segment(A_values, B_values, P)
    # Find the index of the closest point with the minimal distance
    min_distance_index = np.argmin(distances)
   
    # Return the closest point with the minimal distance
    closest_point = closest_points[min_distance_index]
    min_distance = distances[min_distance_index]

    tree = cKDTree(path)
    min_distance_point, index_point = tree.query(P, k=1)
    if min_distance_point<min_distance:
        closest_point=path[index_point]
        min_distance=min_distance_point

    return closest_point, min_distance    

def closest_position_on_path(path,P):
    # Calculate the closest points and distances for each segment
    A_values = path[:-1]  # Starting points of segments
    B_values = path[1:]   # Ending points of segments
    _, distances, factors = closest_point_on_segment(A_values, B_values, P)
    # Find the index of the closest point with the minimal distance
    min_distance_index = np.argmin(distances)
   
    # Return the closest point with the minimal distance
    min_distance = distances[min_distance_index]
    min_factor = factors[min_distance_index]

    tree = cKDTree(path)
    min_distance_point, index_point = tree.query(P, k=1)
    if min_distance_point<min_distance:
        return calculate_positions_3d(path[:index_point+1,0],path[:index_point+1,1],path[:index_point+1,2],np.ones((3)))[-1]
    
    x=calculate_positions_3d(path[:min_distance_index+1,0],path[:min_distance_index+1,1],path[:min_distance_index+1,2],np.ones((3)))[-1]
    x=x+min_factor*np.linalg.norm(A_values[min_distance_index,:]-B_values[min_distance_index,:])

    return x

def define_plane(point1, point2, direction):
    vector = np.array(point2) - np.array(point1)
    normal = np.cross(vector, np.array(direction))
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
    return normal, np.array(point1)

def get_Plane(point1, point2, direction_dv,im_3d,centroid_3d):
    plane_normal, plane_point = define_plane(point1, point2, direction_dv)
    vector_to_point = centroid_3d - plane_point
    projection = vector_to_point - np.dot(vector_to_point, plane_normal) * plane_normal
    center_plane_point = plane_point + projection

    size_2d = (100,np.max(im_3d.shape)+50)

    direction_1 = np.array([0, point2[1]-point1[1], point2[2]-point1[2]])  # You can adjust the x and y components as needed
    dot_product = np.dot(direction_1, plane_normal)
    direction_1 -= dot_product / np.dot(plane_normal, plane_normal) * plane_normal
    direction_1 = direction_1 / np.linalg.norm(direction_1)

    direction_2=np.cross(plane_normal,direction_1)
    direction_2 = direction_2 / np.linalg.norm(direction_2)
    return center_plane_point, direction_2 ,direction_1 , size_2d,plane_normal

def calculate_relative_positions_3d(x_positions, y_positions, z_positions,scales):
    x_diff = np.diff(x_positions)*scales[0]
    y_diff = np.diff(y_positions)*scales[1]
    z_diff = np.diff(z_positions)*scales[2]
    segment_distances = np.sqrt(x_diff*x_diff + y_diff*y_diff+ z_diff*z_diff)
    cumulative_distances = np.concatenate(([0], np.cumsum(segment_distances)))
    s = cumulative_distances / cumulative_distances[-1]
    return s

def interpolate_path(path,s_values, s_position):
    if s_values.shape[0]!=path.shape[0]:
        raise ValueError("s_values and path not same length")


    if s_position < 0 or s_position > 1:
        raise ValueError("s_position should be between 0 and 1")

    num_points = len(path)
    if num_points < 2:
        raise ValueError("Path should have at least two points")

    # Find the index of the rightmost s value less than or equal to s_position
    index = bisect_right(s_values, s_position)

    if index == 0:
        return index,path[0,0],path[0,1],path[0,2]

    if index == s_values.shape[0]:
        return index,path[-1,0],path[-1,1],path[-1,2]

    # Interpolation factor within the segment
    s_start = s_values[index - 1]
    s_end = s_values[index]
    segment_length = s_end - s_start
    interpolation_factor = (s_position - s_start) / segment_length

    # Interpolate the coordinates of the point
    x1, y1, z1 = path[index - 1]
    x2, y2, z2 = path[index]

    interpolated_x = x1 + (x2 - x1) * interpolation_factor
    interpolated_y = y1 + (y2 - y1) * interpolation_factor
    interpolated_z = z1 + (z2 - z1) * interpolation_factor

    return index,interpolated_x, interpolated_y, interpolated_z

def permut_faces(faces):
    lines=set()
    new_faces=[]
    for f_or in faces:
        f=f_or.copy()
        
        while True:
            good=True
            line1=(f[0],f[1])
            line2=(f[1],f[2])
            line3=(f[2],f[0])
            if line1 in lines:
                f=np.random.permutation(f)
                good=False
            if line2 in lines:
                f=np.random.permutation(f)
                good=False
            if line3 in lines:
                f=np.random.permutation(f)
                good=False
            if good:
                break
        lines.add(line1)
        lines.add(line2)
        lines.add(line3)
        new_faces.append(f)
  
    return np.array(new_faces)

def get_coordinate_by_name(df, name):
    row = df[df['name'] == name]
    if not row.empty:
        return row['coordinate_px'].iloc[0]
    else:
        return None

def doIntegration(points,path,coord_1,coord_2,visited,direction_1,direction_2):
    for i in range(len(path)-1):
        if visited[path[i+1]]:
            continue
        dr=points[path[i+1]]-points[path[i]]
        d1idr=np.dot(direction_1[i],dr)
        d2idr=np.dot(direction_2[i],dr)
        d1i1dr=np.dot(direction_1[i+1],dr)
        d2i1dr=np.dot(direction_2[i+1],dr)
        factor_i=np.sqrt(d1idr*d1idr+d2idr*d2idr)
        factor_i1=np.sqrt(d1i1dr*d1i1dr+d2i1dr*d2i1dr)
        dv1=(d1idr/factor_i+d1i1dr/factor_i1)/2
        dv2=(d2idr/factor_i+d2i1dr/factor_i1)/2
        coord_1[path[i+1]]=coord_1[path[i]]+dv1
        coord_2[path[i+1]]=coord_2[path[i]]+dv2

        visited[path[i+1]]=True

def doTransportIntegration(points,path,coord_1,coord_2,visited,direction_1,direction_2,normals):
    for i in range(len(path)-1):
        if visited[path[i+1]]:
            continue
        direction_1[path[i+1]]=direction_1[path[i]]
        direction_2[path[i+1]]=np.cross(normals[path[i+1]],direction_1[path[i+1]])
        direction_2[path[i+1]]=direction_2[path[i+1]]/np.linalg.norm(direction_2[path[i+1]])
        direction_1[path[i+1]]=np.cross(direction_2[path[i+1]],normals[path[i+1]])
        
        dr=points[path[i+1]]-points[path[i]]
        dri=dr-normals[path[i]]*np.dot(dr,normals[path[i]])
        dri1=dr-normals[path[i+1]]*np.dot(dr,normals[path[i+1]])
        d1idr=np.dot(direction_1[path[i]],dri)
        d2idr=np.dot(direction_2[path[i]],dri)
        d1i1dr=np.dot(direction_1[path[i+1]],dri1)
        d2i1dr=np.dot(direction_2[path[i+1]],dri1)
        
        dv1=(d1idr+d1i1dr)/2
        dv2=(d2idr+d2i1dr)/2
        coord_1[path[i+1]]=coord_1[path[i]]+dv1
        coord_2[path[i+1]]=coord_2[path[i]]+dv2
        
        visited[path[i+1]]=True
    return

def create_coord_system(surf_file,Orient_file,Rip_file,scales):
    Orient_df=pd.read_hdf(Orient_file, key='data')
    Rip_df=pd.read_hdf(Rip_file, key='data')
    mesh=pv.read(surf_file)

    view_dir=get_coordinate_by_name(Orient_df,'viewer_direction_DV')*scales
    view_dir=view_dir/np.linalg.norm(view_dir)

    center_Line = np.stack(Rip_df['center_point px'].values)*scales

    #determine center point with direction
    mid_ind=center_Line.shape[0]//2
    center_ind=mesh.find_closest_point(center_Line[mid_ind,:])
    mesh.compute_normals(point_normals=True,cell_normals=False) 
    if np.dot(mesh.point_normals[center_ind],view_dir)>0:
        mesh.flip_normals()
    center_direction=center_Line[mid_ind+1,:]-center_Line[mid_ind-1,:]
    center_direction=center_direction-mesh.point_normals[center_ind]*np.dot(mesh.point_normals[center_ind],center_direction)
    center_direction=center_direction/np.linalg.norm(center_direction)

    direction_1=np.zeros((mesh.points.shape[0],3),dtype=float)
    direction_2=np.zeros((mesh.points.shape[0],3),dtype=float)

    coord_1=np.zeros((mesh.points.shape[0]),dtype=float)
    coord_2=np.zeros((mesh.points.shape[0]),dtype=float)
    visited=np.zeros((mesh.points.shape[0]),dtype=bool)
  
    visited[center_ind]=1
    direction_1[center_ind]=center_direction
    direction_2[center_ind]=np.cross(mesh.point_normals[center_ind],direction_1[center_ind])
    direction_1[center_ind]=np.cross(direction_2[center_ind],mesh.point_normals[center_ind])


    for i in range(mesh.points.shape[0]):
        if i%100==0:
            print(i)
        if visited[i]:
            continue
        try:
            path = mesh.geodesic(center_ind, i).point_data['vtkOriginalPointIds']
        except:
            return None
        doTransportIntegration(mesh.points,path,coord_1,coord_2,visited,direction_1,direction_2,mesh.point_normals)
    coord_1=coord_1-np.min(coord_1)
    
    mesh.point_data['coord_1']=coord_1
    mesh.point_data['coord_2']=coord_2
    mesh.point_data['direction_1']=direction_1
    mesh.point_data['direction_2']=direction_2
    return mesh


def calculateCurvatureTensor(mesh):
    cells = mesh.faces.reshape(-1, 4)[:, 1:]  # This line is adjusted for 'PolyData' objects

    curvature_tensors=np.zeros((mesh.n_points,2,2),dtype=float)

    for i in range(mesh.n_points):
        if i%100==0:
            print(i)
        mask = np.any(cells == i, axis=1)
        cells_including_vertex = cells[mask]
        neighbors = np.unique(cells_including_vertex[cells_including_vertex != i])

        normal1=mesh.point_normals[i]
        point1=mesh.points[i]
        
        nn=len(neighbors)
        curvatures=np.zeros((nn,3),dtype=float)
        directions=np.zeros((nn,3),dtype=float)
        for k,n in enumerate(neighbors):
            normal2=mesh.point_normals[n]
            point2=mesh.points[n]
            dr=point2 - point1
            d = np.linalg.norm(dr)
            curvatures[k] = (normal2-normal1) / d
            directions[k] = dr/d

        direction2d=np.zeros((curvatures.shape[0],2),dtype=float)
        direction2d[:,0]=np.dot(directions,mesh.point_data['direction_1'][i])
        direction2d[:,1]=np.dot(directions,mesh.point_data['direction_2'][i])

        curvature2d=np.zeros((curvatures.shape[0],2),dtype=float)
        curvature2d[:,0]=np.dot(curvatures,mesh.point_data['direction_1'][i])
        curvature2d[:,1]=np.dot(curvatures,mesh.point_data['direction_2'][i])

        N = direction2d.shape[0]
        X_mod = np.zeros((2*N, 3))
        for k in range(N):
            x1, x2 = direction2d[k]
            X_mod[2*k] = [x1, x2, 0]  
            X_mod[2*k + 1] = [0, x1, x2]
        Y_flat = curvature2d.flatten()
        try:
            A_flat, _, _, _ = np.linalg.lstsq(X_mod, Y_flat, rcond=None)
        except:
            return None
        curvature_tensors[i] = np.array([[A_flat[0], A_flat[1]],
                                                    [A_flat[1], A_flat[2]]])

    
    vals,vec=np.linalg.eigh(curvature_tensors)
    
    avg_curvature=(vals[:,0]+vals[:,1])/2
    gauss_curvature=(vals[:,0]*vals[:,1])

    # avg_curvature_pv=mesh.curvature(curv_type='mean')
    # gauss_curvature_pv=mesh.curvature(curv_type='gaussian')

    #import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.scatter(avg_curvature, avg_curvature_pv)
    # plt.title('avg')
    # plt.xlabel('own')
    # plt.ylabel('pv')
    # plt.grid(alpha=0.75)
    # plt.show()
    # plt.figure(figsize=(10, 6))
    # plt.scatter(gauss_curvature, gauss_curvature_pv)
    # plt.title('gauss')
    # plt.xlabel('own')
    # plt.ylabel('pv')
    # plt.grid(alpha=0.75)
    # plt.show()

    # average = np.mean(avg_curvature)
    # std_dev = np.std(avg_curvature)
    # outliers = np.abs(avg_curvature - average) > 2 * std_dev
    # avg_curvature[outliers] = np.nan

    
    # valid_values = avg_curvature[~np.isnan(avg_curvature)]
    # plt.figure(figsize=(10, 6))
    # plt.hist(valid_values, bins=10, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Histogram of Values Distribution')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()

    # average = np.mean(gauss_curvature)
    # std_dev = np.std(gauss_curvature)
    # outliers = np.abs(gauss_curvature - average) > 2 * std_dev
    # gauss_curvature[outliers] = np.nan

    # valid_values = gauss_curvature[~np.isnan(gauss_curvature)]
    # plt.figure(figsize=(10, 6))
    # plt.hist(valid_values, bins=10, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Histogram of Values Distribution')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()


    mesh.point_data['curvature_tensor']=curvature_tensors
    mesh.point_data['main_curvature_directions']=vec
    mesh.point_data['main_curvatures']=vals
    mesh.point_data['avg_curvature']=avg_curvature
    mesh.point_data['gauss_curvature']=gauss_curvature

    return 1

    p = pv.Plotter()
    p.add_mesh(mesh,scalars="avg_curvature", color="grey", show_edges=False)
    p.show()

    p = pv.Plotter()
    p.add_mesh(mesh,scalars="gauss_curvature", color="grey", show_edges=False)
    p.show()


  
    
        

    import plotly.graph_objects as go
    vectors = directions
    fig = go.Figure()
    for vector in vectors:
        fig.add_trace(go.Scatter3d(x=[0, vector[0]],
                                y=[0, vector[1]],
                                z=[0, vector[2]],
                                mode='lines',
                                line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter3d(x=[0, mesh.point_normals[i,0]],
                                y=[0, mesh.point_normals[i,1]],
                                z=[0, mesh.point_normals[i,2]],
                                mode='lines',
                                line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter3d(x=[0, mesh.point_data['direction_1'][i,0]],
                                y=[0, mesh.point_data['direction_1'][i,1]],
                                z=[0, mesh.point_data['direction_1'][i,2]],
                                mode='lines',
                                line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter3d(x=[0, mesh.point_data['direction_2'][i,0]],
                                y=[0, mesh.point_data['direction_2'][i,1]],
                                z=[0, mesh.point_data['direction_2'][i,2]],
                                mode='lines',
                                line=dict(color='green', width=3)))
    fig.update_layout(scene=dict(xaxis_title='X',
                                yaxis_title='Y',
                                zaxis_title='Z',
                                aspectmode='data'),
                    title='Vectors from Origin')
    fig.show()

    return
    return

def evalStatus_Coord(FlatFin_dir_path):
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
        return MetaData

    if not MetaData['Coord_MetaData']['Coord version']==Coord_version:
        return MetaData  
    
    if not MetaData['Coord_MetaData']['input Surface checksum']==MetaData['Surface_MetaData']['output Surface checksum']:
        return MetaData
    
    if not MetaData['Coord_MetaData']['input Orient checksum']==MetaData['Orient_MetaData']['output Orient checksum']:
        return MetaData

    return False

def make_Coord():
    FlatFin_folder_list=os.listdir(FlatFin_path)
    FlatFin_folder_list = [item for item in FlatFin_folder_list if os.path.isdir(os.path.join(FlatFin_path, item))]
    for FlatFin_folder in FlatFin_folder_list:
        print(FlatFin_folder)
        FlatFin_dir_path=os.path.join(FlatFin_path,FlatFin_folder)
 
        PastMetaData=evalStatus_Coord(FlatFin_dir_path)
        if not isinstance(PastMetaData,dict):
            continue

        MetaData_Surface=PastMetaData['Surface_MetaData']
        Surface_file_name=MetaData_Surface['Surface file']
        Rip_file=MetaData_Surface['Rip file']

        MetaData_Orient=PastMetaData['Orient_MetaData']
        Orient_file=MetaData_Orient['Orient file']

        scales=PastMetaData['CenterLine_MetaData']['scales'].copy() 

        #actual calculation
        mesh = create_coord_system(os.path.join(FlatFin_dir_path,Surface_file_name),os.path.join(FlatFin_dir_path,Orient_file),os.path.join(FlatFin_dir_path,Rip_file),scales)
        if mesh is None:
            continue
       
        res=calculateCurvatureTensor(mesh)
        if res is None:
            continue
        
        Surface_file=os.path.join(FlatFin_dir_path,Surface_file_name)
        mesh.save(Surface_file)

        MetaData_Coord={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_Coord['git hash']=sha
        MetaData_Surface['git repo']='Sahrapipline'
        MetaData_Coord['Coord version']=Coord_version
        MetaData_Coord['Surface file']=Surface_file_name
        MetaData_Coord['scales']=PastMetaData['Orient_MetaData']['scales']
        MetaData_Coord['condition']=PastMetaData['Orient_MetaData']['condition']
        MetaData_Coord['time in hpf']=PastMetaData['Surface_MetaData']['time in hpf']
        MetaData_Coord['experimentalist']=PastMetaData['Orient_MetaData']['experimentalist']
        MetaData_Coord['genotype']=PastMetaData['Surface_MetaData']['genotype']
        MetaData_Coord['input Surface checksum']=PastMetaData['Surface_MetaData']['output Surface checksum']
        MetaData_Coord['input Orient checksum']=PastMetaData['Orient_MetaData']['output Orient checksum']
        check_Surface=get_checksum(Surface_file, algorithm="SHA1")
        MetaData_Coord['output Surface checksum']=check_Surface

        writeJSON(FlatFin_dir_path,'Coord_MetaData',MetaData_Coord)
        return

if __name__ == "__main__":
    make_Coord() #construct coord system and curvature

 #mesh.save(os.path.join(LM_dir_path,Surface_file))
#mesh=pv.read(os.path.join(LM_dir_path,Surface_file))

# mesh['vectors'] = mesh.point_normals*5
# mesh.set_active_vectors("vectors")
# p = pv.Plotter()
# p.add_mesh(mesh.arrows, lighting=False, scalar_bar_args={'title': "Vector Magnitude"})
# p.add_mesh(mesh,scalars="coord_1", color="grey", ambient=0.6, opacity=0.5, show_edges=False)
# p.show()