import numpy as np
import cv2
import math
from typing import List
import napari
from scipy.interpolate import UnivariateSpline
import os
import git
import pandas as pd
from simple_file_checksum import get_checksum
from bisect import bisect_right

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from IO import *

        
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
    import math
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
        if intersections.shape[0] > 10:
            all_centroid_x.append(np.mean(intersections[:, 1]))
            all_centroid_y.append(np.mean(intersections[:, 0]))
    res=np.array((all_centroid_y,all_centroid_x)).T
    return res

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

def calculate_relative_positions_3d(x_positions, y_positions, z_positions):
    x_diff = np.diff(x_positions)
    y_diff = np.diff(y_positions)
    z_diff = np.diff(z_positions)
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

def center_line_session(im_file,old_df):
    #print(old_df)
    point1 = old_df[old_df['name'] == 'Proximal_pt']['coordinate_px'].values[0]
    point2 = old_df[old_df['name'] == 'Distal_pt']['coordinate_px'].values[0]
    direction_dv = old_df[old_df['name'] == 'fin_plane']['coordinate_px'].values[0]
    im_3d=getImage(im_file)

    # viewer = napari.Viewer()
    # viewer.add_labels(im_3d, name='Mask')
    # napari.run()

    #centroid_3d = ndimage.center_of_mass(im_3d)
    centroid_3d=np.array(im_3d.shape,dtype=float)/2

    center_plane_point, direction_2 ,direction_1 , size_2d,plane_normal=get_Plane(point1, point2, direction_dv,im_3d,centroid_3d)

    im = create_2d_image_from_3d(center_plane_point,direction_2 ,direction_1 , size_2d, im_3d).astype(int)
    im_mask=im>0

    viewer = napari.Viewer(ndisplay=2)
  
    im_mask = np.array(im_mask, dtype=np.uint8)
    kernel_size = (11, 11) 
    blurred_image = cv2.GaussianBlur(im_mask, kernel_size, 0)
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
        return None
 
    z_3d, y_3d, x_3d=map_2d_to_3d(center_plane_point, direction_1, direction_2, size_2d, resulting_path[:,0],resulting_path[:,1])
    center_line_path_3d = np.column_stack((z_3d, y_3d, x_3d))

    return center_line_path_3d

def evalStatus_center_line(FlatFin_dir_path):
    """
    checks MetaData to desire wether to evaluate the file or not
    returns a dict of MetaData to be written if we should evaluate the file
    """
    
    MetaData=get_JSON(FlatFin_dir_path)
    if not 'Orient_MetaData' in MetaData:
        print('no Orient_MetaData')
        return False

    if not 'CenterLine_MetaData' in MetaData:
        return MetaData

    if not MetaData['CenterLine_MetaData']['CenterLine version']==CenterLine_version:
        return MetaData  
    
    if not MetaData['CenterLine_MetaData']['input Orient checksum']==MetaData['Orient_MetaData']['output Orient checksum']:
        return MetaData

    return False


def make_center_line():
    FlatFin_folder_list=os.listdir(FlatFin_path)
    FlatFin_folder_list = [item for item in FlatFin_folder_list if os.path.isdir(os.path.join(FlatFin_path, item))]
    for FlatFin_folder in FlatFin_folder_list:
        print(FlatFin_folder)
        FlatFin_dir_path=os.path.join(FlatFin_path,FlatFin_folder)

        PastMetaData=evalStatus_center_line(FlatFin_dir_path)
        if not isinstance(PastMetaData,dict):
            continue
        data_name=FlatFin_folder[:-len('_FlatFin')]


        Orient_file=os.path.join(FlatFin_dir_path,PastMetaData['Orient_MetaData']['Orient file'])
        Orient_df=pd.read_hdf(Orient_file, key='data')

        #actual calculation
        print('start interactive session')
        center_line_path_3d=center_line_session(os.path.join(finmasks_path,data_name,data_name+'.tif'),Orient_df)
        if(center_line_path_3d is None):
            continue
        #print(center_line_path_3d)

        CenterLine_file_name=data_name+'_CenterLine.npy'
        CenterLine_file=os.path.join(FlatFin_dir_path,CenterLine_file_name)
        saveArr(center_line_path_3d,CenterLine_file)

        MetaData_CenterLine={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_CenterLine['git hash']=sha
        MetaData_CenterLine['git repo']='Sahrapipline'
        MetaData_CenterLine['CenterLine version']=CenterLine_version
        MetaData_CenterLine['CenterLine file']=CenterLine_file_name
        MetaData_CenterLine['scales ZYX']=PastMetaData['Orient_MetaData']['scales ZYX']
        MetaData_CenterLine['condition']=PastMetaData['Orient_MetaData']['condition']
        MetaData_CenterLine['time in hpf']=PastMetaData['Orient_MetaData']['time in hpf']
        MetaData_CenterLine['experimentalist']=PastMetaData['Orient_MetaData']['experimentalist']
        MetaData_CenterLine['genotype']=PastMetaData['Orient_MetaData']['genotype']
        MetaData_CenterLine['input Orient checksum']=PastMetaData['Orient_MetaData']['output Orient checksum']
        check_LM=get_checksum(CenterLine_file, algorithm="SHA1")
        MetaData_CenterLine['output CenterLine checksum']=check_LM
        writeJSON(FlatFin_dir_path,'CenterLine_MetaData',MetaData_CenterLine)


if __name__ == "__main__":
    make_center_line() 
