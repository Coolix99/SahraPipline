import numpy as np
from typing import List
import napari
import os
import cv2
import git
import pandas as pd
from simple_file_checksum import get_checksum
import re
from scipy.ndimage import label
from scipy import ndimage
from skimage import measure
import pyvista as pv
import math
from scipy.interpolate import UnivariateSpline
import pymeshfix as mf
from scipy.spatial import cKDTree
from bisect import bisect_right

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

def map_2d_to_3d(center_plane_point, direction_1, direction_2, size_2d, y_2d, x_2d):
    z_3d = center_plane_point[0] + direction_1[0]*(x_2d-size_2d[1]/2) + direction_2[0]*(y_2d-size_2d[0]/2)
    y_3d = center_plane_point[1] + direction_1[1]*(x_2d-size_2d[1]/2) + direction_2[1]*(y_2d-size_2d[0]/2)
    x_3d = center_plane_point[2] + direction_1[2]*(x_2d-size_2d[1]/2) + direction_2[2]*(y_2d-size_2d[0]/2)

    return z_3d, y_3d, x_3d

def create_2d_image_from_3d(center_plane_point,direction_2 ,direction_1 , size_2d,  image_3d):
    y_2d, x_2d = np.meshgrid(np.arange(size_2d[0]) , 
                             np.arange(size_2d[1]) , indexing='ij')

    z_3d, y_3d, x_3d  = map_2d_to_3d(center_plane_point, direction_1, direction_2, size_2d, y_2d, x_2d)
    
    z_3d = np.round(z_3d).astype(int)
    y_3d = np.round(y_3d).astype(int)
    x_3d = np.round(x_3d).astype(int)

    # Set positions outside the range to zero
    mask = (z_3d >= 0) & (z_3d < image_3d.shape[0]) & \
           (y_3d >= 0) & (y_3d < image_3d.shape[1]) & \
           (x_3d >= 0) & (x_3d < image_3d.shape[2])

    image_2d = np.zeros(size_2d)
    image_2d[mask] = image_3d[z_3d[mask], y_3d[mask], x_3d[mask]]

    return image_2d

def vertical_center_line(prox_dist_pos,closed_image):
    start_vertical_position = prox_dist_pos
    end_vertical_position = np.max(np.where(np.any(closed_image > 0, axis=0)))

    all_centroid_x=[]
    all_centroid_y=[]
    for x_pos in range(int(start_vertical_position),end_vertical_position):
        horizontal_lines = closed_image[:,x_pos]
        mean_positions = np.mean(np.where(horizontal_lines > 0))
        if math.isnan(mean_positions):
            continue
        all_centroid_y.append(mean_positions)
        all_centroid_x.append(x_pos)
    
    res=np.array((all_centroid_y,all_centroid_x)).T
    return res

def construct_Surface(im_file,old_df,center_line_path_3d,scales):
    point1 = old_df[old_df['name'] == 'Proximal_pt']['coordinate_px'].values[0]
    point2 = old_df[old_df['name'] == 'Distal_pt']['coordinate_px'].values[0]
    direction_dv = old_df[old_df['name'] == 'fin_plane']['coordinate_px'].values[0]
    im_3d=im_file
    centroid_3d = ndimage.center_of_mass(im_3d)

    center_plane_point, direction_2 ,direction_1 , size_2d,plane_normal=get_Plane(point1, point2, direction_dv,im_3d,centroid_3d)

    t=calculate_relative_positions_3d(center_line_path_3d[:,0],center_line_path_3d[:,1],center_line_path_3d[:,2],scales)

    all_lines_3d=[]
    all_lines_PD=[]
    all_lines_centerPoint=[]
    all_lines_AP=[]
    direction_1[:]=plane_normal[:]
    rel_position=np.linspace(0,1,100)
    for i in range(rel_position.shape[0]):
        print(i)
        ind,x,y,z=interpolate_path(center_line_path_3d,t,rel_position[i])
        center_plane_point=np.array((x,y,z))
        if(ind==center_line_path_3d.shape[0]):
            plane_normal=center_line_path_3d[ind-2,:]-center_line_path_3d[ind-1,:]
        else:
            plane_normal=center_line_path_3d[ind-1,:]-center_line_path_3d[ind,:]
        plane_normal=plane_normal/np.linalg.norm(plane_normal)
        direction_2=np.cross(direction_1,plane_normal)
        
        im = create_2d_image_from_3d(center_plane_point,direction_2 ,direction_1 , size_2d, im_3d)

        im = np.array(im, dtype=np.uint8)
        kernel_size = (11, 11) 
        blurred_image = cv2.GaussianBlur(im, kernel_size, 0)
        threshold_value = np.max(blurred_image)/2
        _, binary_image = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        non_zero_pixels = cv2.findNonZero(closed_image)
        x, y, width, height = cv2.boundingRect(non_zero_pixels)
        if max(width,height)<10:
                continue

        path_data_ver=vertical_center_line(0,closed_image)

        z_3d, y_3d, x_3d=map_2d_to_3d(center_plane_point, direction_1, direction_2, size_2d, path_data_ver[:,0],path_data_ver[:,1])
        line_path_3d = np.column_stack((z_3d, y_3d, x_3d))
        all_lines_3d.append(line_path_3d)
        all_lines_PD.append(rel_position[i])
        all_lines_centerPoint.append(center_plane_point)

        positions_AP=calculate_positions_3d(z_3d, y_3d, x_3d,scales)
        mid_pos=closest_position_on_path(line_path_3d*scales,center_plane_point*scales)
        #print(mid_pos)
        positions_AP=positions_AP-mid_pos 
        all_lines_AP.append(positions_AP)

    L=calculate_positions_3d(center_line_path_3d[:,0],center_line_path_3d[:,1],center_line_path_3d[:,2],scales)[-1]
    all_lines_PD=np.array(all_lines_PD-all_lines_PD[0])*L
   
    # viewer = napari.Viewer(ndisplay=3)
    # im_layer = viewer.add_labels(im_3d)
    # for i in range(len(all_lines_3d)):
    #     viewer.add_shapes(all_lines_3d[i], shape_type='path', edge_color='green', edge_width=2)
    # napari.run()
      
    data = {'path px': all_lines_3d, 
            'PD_position mum': all_lines_PD,
            'center_point px': all_lines_centerPoint,
            'AP_position mum':all_lines_AP}
    

    df = pd.DataFrame(data)
    print(df)
    print(df['AP_position mum'])
    get_first_element = lambda x: x[0]
    min_first_element = df['AP_position mum'].apply(get_first_element).min()
    print("Min of the first elements:", min_first_element)
    get_last_element = lambda x: x[-1]
    max_last_element = df['AP_position mum'].apply(get_last_element).max()
    print("Maximum of the last elements:", max_last_element)
    
    P=df['path px'][0]
    P_mum=P*scales
    P_coord=np.zeros((P.shape[0],2))
    P_coord[:,0]=df['AP_position mum'][0]
    P_coord[:,1]=df['PD_position mum'][0]
    for i in range(1,df.shape[0]):
        p=df['path px'][i]
        P=np.concatenate((P,p),axis=0)
        P_mum=np.concatenate((P_mum,p*scales),axis=0)
        p_coord=np.zeros((p.shape[0],2))
        p_coord[:,0]=df['AP_position mum'][i]
        p_coord[:,1]=df['PD_position mum'][i]
        P_coord=np.concatenate((P_coord,p_coord),axis=0)
    
    cloud = pv.PolyData(P_mum)
    surf = cloud.delaunay_2d(tol=0.01,alpha=20)

    meshfix = mf.MeshFix(surf.triangulate())
    holes = meshfix.extract_holes()
    indices1 = np.arange(len(holes.lines)) % 3 == 1
    indices2 = np.arange(len(holes.lines)) % 3 == 2
    v1=holes.lines[indices1]
    v2=holes.lines[indices2]
    result=np.zeros((v1.shape[0],2),dtype=np.uint)
    result[:,0] = v1
    result[:,1] = v2

    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(result)
    cycles = list(nx.simple_cycles(G))
    small_cycles = [cycle for cycle in cycles if len(cycle) < 20]
    for cycle in small_cycles:
        point_indices = []
        for i in range(holes.points[cycle].shape[0]):
            idx = surf.find_closest_point(holes.points[cycle][i])
            point_indices.append(idx)
        point_indices_np=np.array((len(point_indices),*point_indices),dtype=np.uint)
        surf.faces=np.concatenate((surf.faces,point_indices_np),dtype=int)
    surf=surf.triangulate()

    smooth_cube = surf.smooth(1000, feature_smoothing=False,boundary_smoothing=False)
    smooth_cube = smooth_cube.clean()

    tree = cKDTree(P_mum)
    point_coord=np.zeros((smooth_cube.points.shape[0],2))
    for i in range(smooth_cube.points.shape[0]):
        distances, indices = tree.query(smooth_cube.points[i], k=3)
        if distances[0] == 0:
            point_coord[i, :] = P_coord[indices[0], :]
            continue
        inverse_distances = 1.0 / distances
        interpolated_value = np.sum(P_coord[indices, :] * inverse_distances[:, np.newaxis], axis=0) / np.sum(inverse_distances)
        point_coord[i, :] = interpolated_value
    smooth_cube.point_data['AP mum']=point_coord[:,0]
    smooth_cube.point_data['PD mum']=point_coord[:,1]
    smooth_cube.point_data['Coord px']=smooth_cube.points/scales

    # p = pv.Plotter()
    # p.add_mesh(smooth_cube,scalars="AP mum")
    # p.show()
    # p = pv.Plotter()
    # p.add_mesh(smooth_cube,scalars="PD mum")
    # p.show()

    return smooth_cube,df
        
def define_plane(point1, point2, direction):
    v = np.array(point2) - np.array(point1)
    normal = np.cross(v, np.array(direction))
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector
    return normal, np.array(point1)

def map_2d_to_3d(center_plane_point, direction_1, direction_2, size_2d, y_2d, x_2d):
    z_3d = center_plane_point[0] + direction_1[0]*(x_2d-size_2d[1]/2) + direction_2[0]*(y_2d-size_2d[0]/2)
    y_3d = center_plane_point[1] + direction_1[1]*(x_2d-size_2d[1]/2) + direction_2[1]*(y_2d-size_2d[0]/2)
    x_3d = center_plane_point[2] + direction_1[2]*(x_2d-size_2d[1]/2) + direction_2[2]*(y_2d-size_2d[0]/2)

    return z_3d, y_3d, x_3d

def create_2d_image_from_3d(center_plane_point,direction_2 ,direction_1 , size_2d,  image_3d):
    y_2d, x_2d = np.meshgrid(np.arange(size_2d[0]) , 
                             np.arange(size_2d[1]) , indexing='ij')

    z_3d, y_3d, x_3d  = map_2d_to_3d(center_plane_point, direction_1, direction_2, size_2d, y_2d, x_2d)
    
    z_3d = np.round(z_3d).astype(int)
    y_3d = np.round(y_3d).astype(int)
    x_3d = np.round(x_3d).astype(int)

    # Set positions outside the range to zero
    mask = (z_3d >= 0) & (z_3d < image_3d.shape[0]) & \
           (y_3d >= 0) & (y_3d < image_3d.shape[1]) & \
           (x_3d >= 0) & (x_3d < image_3d.shape[2])

    image_2d = np.zeros(size_2d)
    image_2d[mask] = image_3d[z_3d[mask], y_3d[mask], x_3d[mask]]

    return image_2d

def radial_center_line(prox_dist_pos,y_pos,closed_image):
    outside_point = (prox_dist_pos,y_pos )  

    num_angles = 91  # Adjust as needed
    angles = np.linspace(0, math.pi / 2, num=num_angles)

    all_centroid_x=[]
    all_centroid_y=[]

    for i, angle in enumerate(angles):
        line_length = max(closed_image.shape) * 1.5  # Adjust as needed
        end_x = outside_point[0] - line_length * math.cos(angle)
        end_y = outside_point[1] - line_length * math.sin(angle)

        # Create an array of coordinates along the line from outside_point to the endpoint
        x_coords = np.linspace(outside_point[0], end_x, num=int(line_length))
        y_coords = np.linspace(outside_point[1], end_y, num=int(line_length))
        line_coords = np.column_stack((y_coords,x_coords)).astype(int)

        # Ensure that coordinates are within image bounds
        valid_coords_mask = np.logical_and.reduce(
            (line_coords[:, 0] >= 0, line_coords[:, 0] < closed_image.shape[0],
            line_coords[:, 1] >= 0, line_coords[:, 1] < closed_image.shape[1]))
        # Extract valid coordinates
        valid_line_coords = line_coords[valid_coords_mask]
        # Find intersections with the binary image
        intersections_mask = closed_image[valid_line_coords[:, 0], valid_line_coords[:, 1]] > 0
        # Get intersection points
        intersections = valid_line_coords[intersections_mask]
        if intersections.shape[0] > 1:
            all_centroid_x.append(np.mean(intersections[:, 1]))
            all_centroid_y.append(np.mean(intersections[:, 0]))
    res=np.array((all_centroid_y,all_centroid_x)).T
    return res


def get_Plane(point1, point2, direction_dv,im_3d,centroid_3d):
    plane_normal, plane_point = define_plane(point1, point2, direction_dv)
    vector_to_point = centroid_3d - plane_point
    projection = vector_to_point - np.dot(vector_to_point, plane_normal) * plane_normal
    center_plane_point = plane_point + projection
    print(im_3d.shape)
    size_2d = (im_3d.shape[0]+20,np.max(im_3d.shape)+200)
    print(size_2d)
    direction_1 = np.array([0, point2[1]-point1[1], point2[2]-point1[2]])  # You can adjust the x and y components as needed
    dot_product = np.dot(direction_1, plane_normal)
    direction_1 -= dot_product / np.dot(plane_normal, plane_normal) * plane_normal
    direction_1 = direction_1 / np.linalg.norm(direction_1)

    direction_2=np.cross(plane_normal,direction_1)
    direction_2 = direction_2 / np.linalg.norm(direction_2)
    return center_plane_point, direction_2 ,direction_1 , size_2d,plane_normal

def calculate_relative_positions(x_positions, y_positions):
    x_diff = np.diff(x_positions)
    y_diff = np.diff(y_positions)
    segment_distances = np.sqrt(x_diff**2 + y_diff**2)
    
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

def center_line_session(flipped_mask,old_df):
    point1 = old_df[old_df['name'] == 'Proximal_pt']['coordinate_px'].values[0]
    point2 = old_df[old_df['name'] == 'Distal_pt']['coordinate_px'].values[0]
    direction_dv = old_df[old_df['name'] == 'fin_plane']['coordinate_px'].values[0]
    im_3d=flipped_mask
    #centroid_3d = ndimage.center_of_mass(im_3d)
    centroid_3d=np.array(im_3d.shape,dtype=float)/2

    center_plane_point, direction_2 ,direction_1 , size_2d,plane_normal=get_Plane(point1, point2, direction_dv,im_3d,centroid_3d)
    print(center_plane_point)
    im = create_2d_image_from_3d(center_plane_point,direction_2 ,direction_1 , size_2d, im_3d)

    viewer = napari.Viewer(ndisplay=2)
  
    im = np.array(im, dtype=np.uint8)
    kernel_size = (11, 11) 
    blurred_image = cv2.GaussianBlur(im, kernel_size, 0)
    threshold_value = np.max(blurred_image)/2
    _, binary_image = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    last_pos=None
    last_viewer_direction=None
    prox_dist_pos = None
    interpolated_data=None
    path_layer=None

    non_zero_pixels = cv2.findNonZero(closed_image)
    x, y, width, height = cv2.boundingRect(non_zero_pixels)
    y_pos=y+height-10 
    line_data=np.array([[y_pos,0],[y_pos,im.shape[1]-1]])
    viewer.add_shapes(data=line_data,shape_type='line',edge_color='red',edge_width=2)

    im_layer = viewer.add_labels(im)

    @im_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        nonlocal last_pos,last_viewer_direction
        near_point, far_point = layer.get_ray_intersections(
            event.position,
            event.view_direction,
            event.dims_displayed
        )
        last_pos=event.position
        last_viewer_direction=event.view_direction
        print(event.position,
            event.view_direction,
            event.dims_displayed)


    @viewer.bind_key('a')
    def add_point_a(viewer):
        nonlocal last_pos, prox_dist_pos, y_pos, interpolated_data,path_layer
        prox_dist_pos=last_pos[1]
        if last_pos is not None:
            line_data=np.array([[0,prox_dist_pos],[im.shape[0]-1,prox_dist_pos]])
            viewer.add_shapes(data=line_data,shape_type='line',edge_color='green',edge_width=2)

            path_data_rad=radial_center_line(prox_dist_pos,y_pos,closed_image)
            path_data_ver=vertical_center_line(prox_dist_pos,closed_image)
            path_data=np.concatenate((path_data_rad, path_data_ver), axis=0)
            viewer.add_shapes(path_data, shape_type='path', edge_color='red', edge_width=2)
            
            s=calculate_relative_positions(path_data[:,0],path_data[:,1])
            sp0=UnivariateSpline(s, path_data[:,0],k=3, s=200)
            sp1=UnivariateSpline(s, path_data[:,1],k=3, s=200)
            t=np.linspace(0, 1, 100)
            interpolated_data=np.zeros((t.shape[0]+2,2))
            interpolated_data[1:-1,0]=sp0(t)
            interpolated_data[1:-1,1]=sp1(t)
            v=interpolated_data[2,:]-interpolated_data[1,:]
            v=v/np.linalg.norm(v)
            interpolated_data[0,:]=interpolated_data[1,:]-v*100
            v=interpolated_data[-2,:]-interpolated_data[-3,:]
            v=v/np.linalg.norm(v)
            interpolated_data[-1,:]=interpolated_data[-2,:]+v*100

            path_layer=viewer.add_shapes(interpolated_data, shape_type='path', edge_color='green', edge_width=2)

    napari.run()
    try:
        resulting_path = path_layer.data[0] 
    except:
        print('Fatal error')
        return None
 
    z_3d, y_3d, x_3d=map_2d_to_3d(center_plane_point, direction_1, direction_2, size_2d, resulting_path[:,0],resulting_path[:,1])
    center_line_path_3d = np.column_stack((z_3d, y_3d, x_3d))

    return center_line_path_3d

def extract_coordinate(df, name):
    row = df[df['name'] == name]
    if not row.empty:
        return row['coordinate_px'].iloc[0]
    else:
        return None

def orient_session(im):
    viewer = napari.Viewer(ndisplay=3)
    im_layer = viewer.add_labels(im)
    last_pos=None
    last_viewer_direction=None
    points = []
    points_data=None
    line_layer=None

    @im_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        nonlocal last_pos,last_viewer_direction
        near_point, far_point = layer.get_ray_intersections(
            event.position,
            event.view_direction,
            event.dims_displayed
        )
        last_pos=event.position
        last_viewer_direction=event.view_direction
        print(event.position,
            event.view_direction,
            event.dims_displayed)
        
    @viewer.bind_key('a')
    def first(viewer):
        nonlocal points,last_pos
        points=[last_pos]

    @viewer.bind_key('b')
    def second(viewer):
        nonlocal points,last_pos,points_data,line_layer
        points.append(last_pos)
        line = np.array([points])
        print(line)
        print(viewer.camera)
        print(viewer)
        try:
            line_layer=viewer.add_shapes(line, shape_type='line', edge_color='red', edge_width=2)
        except:
            print('catched')

        viewer.layers.select_previous()
        points_data = [
        {'coordinate_px': np.array(points[0]), 'name': 'Proximal_pt'},
        {'coordinate_px': np.array(points[1]), 'name': 'Distal_pt'},
        {'coordinate_px': last_viewer_direction, 'name': 'viewer_direction_DV'}
        ]

    @viewer.bind_key('c')
    def first2(viewer):
        nonlocal points,last_pos
        points=[last_pos]

    @viewer.bind_key('d')
    def second2(viewer):
        nonlocal points,last_pos,points_data,line_layer
        points.append(last_pos)
        line = np.array([points])
        try:
            line_layer=viewer.add_shapes(line, shape_type='line', edge_color='green', edge_width=2)
        except:
            print('catched')
        viewer.layers.select_previous()
        viewer.layers.select_previous()
        points_data =points_data+ [
        {'coordinate_px': np.array(points[0]), 'name': 'Anterior_pt'},
        {'coordinate_px': np.array(points[1]), 'name': 'Posterior_pt'},
        {'coordinate_px': last_viewer_direction, 'name': 'viewer_direction_DP'}
        ]

    @viewer.bind_key('e')
    def first3(viewer):
        nonlocal points,last_pos
        points=[last_pos]

    @viewer.bind_key('f')
    def second3(viewer):
        nonlocal points,last_pos,points_data,line_layer
        points.append(last_pos)
        line = np.array([points])
        try:
            line_layer=viewer.add_shapes(line, shape_type='line', edge_color='blue', edge_width=2)
        except:
            print('catched')
        points_data =points_data+ [
        {'coordinate_px': np.array(points[0]), 'name': 'Proximal2_pt'},
        {'coordinate_px': np.array(points[1]), 'name': 'Distal2_pt'},
        {'coordinate_px': last_viewer_direction, 'name': 'viewer_direction_AP'}
        ]
    napari.run()
    df = pd.DataFrame(points_data)
    
    print(df)


    v1=extract_coordinate(df,'Proximal_pt')-extract_coordinate(df,'Distal_pt')
    v1 = v1 / np.linalg.norm(v1)
    v2=extract_coordinate(df,'Anterior_pt')-extract_coordinate(df,'Posterior_pt')
    #v2 = v2 - np.dot(v1, v2)
    v3=extract_coordinate(df,'Proximal2_pt')-extract_coordinate(df,'Distal2_pt')
    n=np.cross(v3,v2)
    n = n / np.linalg.norm(n)

    new_row = pd.DataFrame({'coordinate_px': [n], 'name': ['fin_plane']})
    df = pd.concat([df, new_row], ignore_index=True)

    return df

def extract_condition(s):
    # Using regular expressions to find 'dev' or 'reg'
    match = re.search(r'(dev|reg)', s)
    if match:
        return match.group(1)
    else:
        # If neither 'dev' nor 'reg' is found, raise an error
        raise ValueError(f"No 'dev' or 'reg' found in string: {s}")

def extract_hpf(s):
    """
    Extracts an integer immediately preceding 'hpf' in a given string.

    Args:
    s (str): The input string from which to extract the number.

    Returns:
    int: The extracted integer, or None if no such pattern exists.
    """
    match = re.search(r'(\d+)hpf', s)
    return int(match.group(1)) if match else None

def find_correct_orientation(signal, mask):
    """
    Flips the mask along different axes to ensure all non-zero pixels of the signal image are inside the mask.

    Args:
    signal (np.array): The signal image array.
    mask (np.array): The mask image array.

    Returns:
    np.array: The correctly oriented mask or None if no orientation meets the condition.
    """
    # Check each possible flip configuration

    ind_sig=np.where((signal > 0))
    for axis in [None, 0, 1, (0, 1)]:
        flipped_mask = np.flip(mask, axis=axis) if axis is not None else mask.copy()
        print(axis)
        # Check if all non-zero pixels of the signal are within the mask
        if (np.sum(flipped_mask[ind_sig]>1)/ind_sig[0].shape[0])>0.9:
            return flipped_mask  # Return the correctly oriented mask

    return None  # Return None if no valid configuration is found

def extract_largest_component(mask):
    """
    Extracts the largest connected component from a binary mask.

    Args:
    mask (np.array): The binary mask array.

    Returns:
    np.array: A binary mask of the largest connected component.
    """
    # Label all connected components
    labeled_mask, num_features = label(mask)
    
    # Find the sizes of all components
    sizes = np.bincount(labeled_mask.ravel())
    
    # Determine the label of the largest component (excluding the background, which is label 0)
    largest_label = sizes[1:].argmax() + 1
    
    # Create a mask for the largest connected component
    largest_component = (labeled_mask == largest_label)
    
    return largest_component



def evalStatus_mesh(res_path):
    MetaData=get_JSON(res_path)
    if not 'Mesh_MetaData' in MetaData:
        return MetaData

    if not MetaData['Mesh_MetaData']['Mesh version']==Mesh_version:
        return MetaData  
    
    return False

def create_mesh():
    masks_dir_path=os.path.join(EpFlat_path,'peeled_mask')
    signal_dir_path=os.path.join(EpFlat_path,'signal_upperlayer')
    mask_list=os.listdir(masks_dir_path)
    for mask in mask_list:
        data_name=mask[:-11]
        print(data_name)
        signal_path=os.path.join(signal_dir_path,data_name+'.tif')
        if not os.path.exists(signal_path):
            print('no sig')
            continue
        mask_path=os.path.join(masks_dir_path,mask)

        res_path=os.path.join(EpSeg_path,data_name)
        PastMetaData=evalStatus_mesh(res_path)
        if not isinstance(PastMetaData,dict):
            continue
        make_path(res_path)

        mask_img,scale=getImage_Meta(mask_path)
        sig_img=getImage(signal_path)

        condition=extract_condition(data_name)
        time=extract_hpf(data_name)

        flipped_mask=find_correct_orientation(sig_img, mask_img)
        flipped_mask=extract_largest_component(flipped_mask)

        Orient_df=orient_session(flipped_mask)
        center_line_path_3d=center_line_session(flipped_mask,Orient_df)
        mesh,_=construct_Surface(flipped_mask,Orient_df,center_line_path_3d,scale)

 
        mesh = mesh.decimate(0.5)
        print('decimate')
        boundary = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,non_manifold_edges=False,clear_data=True)
        boundary.plot(color='red', line_width=2, show_edges=True)
        mesh=mesh.subdivide_adaptive(max_edge_len=20.0)
        print('subdev')
        boundary = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,non_manifold_edges=False,clear_data=True)
        boundary.plot(color='red', line_width=2, show_edges=True)
        mesh=mesh.clean()
        print('clean')
        boundary = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,non_manifold_edges=False,clear_data=True)
        boundary.plot(color='red', line_width=2, show_edges=True)
        mesh.compute_normals(point_normals=True, cell_normals=True, auto_orient_normals=True, flip_normals=False)
        mesh.plot(show_edges=True)

        mesh_file_name=data_name+'_mesh.vtk'
        mesh_file=os.path.join(res_path,mesh_file_name)
        mesh.save(mesh_file)
            
        MetaData_Mesh={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_Mesh['git hash']=sha
        MetaData_Mesh['git repo']='Sahrapipline'
        MetaData_Mesh['Mesh version']=Mesh_version
        MetaData_Mesh['Mesh file']=mesh_file_name
        MetaData_Mesh['scales']=[scale[0],scale[1],scale[2]]
        MetaData_Mesh['condition']=condition
        MetaData_Mesh['time in hpf']=time
        MetaData_Mesh['experimentalist']='Sahra'
        MetaData_Mesh['genotype']='WT'
        check_mesh=get_checksum(mesh_file, algorithm="SHA1")
        MetaData_Mesh['output mesh checksum']=check_mesh
        writeJSON(res_path,'Mesh_MetaData',MetaData_Mesh)       

if __name__ == "__main__":
    create_mesh()
