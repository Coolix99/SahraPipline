import numpy as np
from typing import List
import napari
import os
import git
from simple_file_checksum import get_checksum
import pyvista as pv

from config import *
from IO import *



def getIntersections(vol_img,r0,normal,max_value=100,precision=1):
    a = 0
    b = max_value

    coord=np.clip(np.array((int(r0[0]+0.5+normal[0]*a),int(r0[1]+0.5+normal[1]*a),int(r0[2]+0.5+normal[2]*a)),dtype=int),[0,0,0],[vol_img.shape[0]-1,vol_img.shape[1]-1,vol_img.shape[2]-1])
    f_a=vol_img[coord[0],coord[1],coord[2]]
    if not f_a:
        return None

    while abs(b - a) > precision:
        midpoint = (a + b) / 2.0
        coord=np.clip(np.array((int(r0[0]+0.5+normal[0]*a),int(r0[1]+0.5+normal[1]*a),int(r0[2]+0.5+normal[2]*a)),dtype=int),[0,0,0],[vol_img.shape[0]-1,vol_img.shape[1]-1,vol_img.shape[2]-1])
        f_a=vol_img[coord[0],coord[1],coord[2]]
        coord=np.clip(np.array((int(r0[0]+0.5+normal[0]*midpoint),int(r0[1]+0.5+normal[1]*midpoint),int(r0[2]+0.5+normal[2]*midpoint)),dtype=int),[0,0,0],[vol_img.shape[0]-1,vol_img.shape[1]-1,vol_img.shape[2]-1])
        f_mid=vol_img[coord[0],coord[1],coord[2]]
        if f_mid == f_a:
            a = midpoint
        else:
            b = midpoint
    best_upper = (a + b) / 2.0

    a = 0
    b = -max_value
    while abs(b - a) > precision:
        midpoint = (a + b) / 2.0
        coord=np.clip(np.array((int(r0[0]+0.5+normal[0]*a),int(r0[1]+0.5+normal[1]*a),int(r0[2]+0.5+normal[2]*a)),dtype=int),[0,0,0],[vol_img.shape[0]-1,vol_img.shape[1]-1,vol_img.shape[2]-1])
        f_a=vol_img[coord[0],coord[1],coord[2]]
        coord=np.clip(np.array((int(r0[0]+0.5+normal[0]*midpoint),int(r0[1]+0.5+normal[1]*midpoint),int(r0[2]+0.5+normal[2]*midpoint)),dtype=int),[0,0,0],[vol_img.shape[0]-1,vol_img.shape[1]-1,vol_img.shape[2]-1])
        f_mid=vol_img[coord[0],coord[1],coord[2]]
        if f_mid == f_a:
            a = midpoint
        else:
            b = midpoint
    best_lower = (a + b) / 2.0

    return r0+normal*best_upper, r0+normal*best_lower

def calculate_Thickness(im_file,surf_file,scales):
    vol_img=getImage(im_file)
    mesh=pv.read(surf_file)
    import skimage as ski
    vol_img=ski.filters.gaussian(vol_img, sigma=(2,5,5),truncate=3)>0.5
    points_px=mesh.point_data['Coord px']
    normals_px=mesh.point_normals/scales

   
    upper_dist=np.zeros(points_px.shape[0],dtype=float)
    lower_dist=np.zeros(points_px.shape[0],dtype=float)
    for i in range(points_px.shape[0]):
        res=getIntersections(vol_img,points_px[i,:],normals_px[i,:])
        if res is None:
            upper_dist[i]=0
            lower_dist[i]=0
            continue
        upper_point, lower_point=res
        #print(upper_point,lower_point)
        upper_dist[i]=np.linalg.norm((upper_point-points_px[i,:])*scales)
        lower_dist[i]=np.linalg.norm((lower_point-points_px[i,:])*scales)
        
    mesh.point_data['upper_dist']=upper_dist
    mesh.point_data['lower_dist']=lower_dist
    mesh.point_data['thickness']=upper_dist+lower_dist

    # viewer = napari.Viewer(ndisplay=3)
    # faces = mesh.faces.reshape(-1, 4)[:, 1:]
    # surface = (points_px, faces)
    # viewer.add_surface(surface)
    # viewer.add_labels(vol_img)
    # #viewer.add_labels(blur)
    # viewer.add_points(points_plot)
    # napari.run()

    # p = pv.Plotter()
    # p.add_mesh(mesh,scalars="thickness", color="grey", ambient=0.6, opacity=0.5, show_edges=False)
    # p.show()
    # p = pv.Plotter()
    # p.add_mesh(mesh,scalars="lower_dist", color="grey", ambient=0.6, opacity=0.5, show_edges=False)
    # p.show()
    # p = pv.Plotter()
    # p.add_mesh(mesh,scalars="diff", color="grey", ambient=0.6, opacity=0.5, show_edges=False)
    # p.show()

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.hist(upper_dist-lower_dist, bins=30, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Histogram of Values Distribution')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()
    return mesh

def evalStatus_Thickness(image_dir_path,LMcoord_dir_path):
    AllMetaData_image=get_JSON(image_dir_path)
    if not 'vol_image_MetaData' in AllMetaData_image:
        print('no vol_image_MetaData -> skip')
        return False

    if not isinstance(AllMetaData_image['vol_image_MetaData'],dict):
        print('vol MetaData not good')
        return False
    
    res={}
    res['vol_image_MetaData']=AllMetaData_image['vol_image_MetaData']

    AllMetaData_LM=get_JSON(LMcoord_dir_path)

    if not 'Surface_MetaData' in AllMetaData_LM:
        print('no Surface_MetaData ->skip')
        return False
    if not 'Coord_MetaData' in AllMetaData_LM:
        print('no Coord_MetaData ->skip')
        return False

    res['CenterLine_MetaData']=AllMetaData_LM['CenterLine_MetaData']
    res['Orient_MetaData']=AllMetaData_LM['Orient_MetaData']
    res['nuclei_image_MetaData']=AllMetaData_LM['nuclei_image_MetaData']
    res['Surface_MetaData']=AllMetaData_LM['Surface_MetaData']
    res['Coord_MetaData']=AllMetaData_LM['Coord_MetaData']
    try:
        if AllMetaData_LM['Orient_MetaData']['Orient version']!=Orient_version:
            print('wrong version Orient')
            return False
        if AllMetaData_LM['CenterLine_MetaData']['CenterLine version']!=CenterLine_version:
            print('wrong version CL')
            return False
        if AllMetaData_LM['Coord_MetaData']['Coord version']!=Coord_version:
            print('wrong version Coord')
            return False
        if AllMetaData_LM['Thickness_MetaData']['Thickness version']!=Thickness_version:
            print('wrong version Thickness')
            return res
        if AllMetaData_LM['Thickness_MetaData']['input Surface checksum']!=AllMetaData_LM['Coord_MetaData']['output Surface checksum']:
            print('different Input Coord')
            return res
        
        return False #already done
    except:
        return res #probably no Coord_MetaData

def get_name(input_string):
    if input_string.endswith("_vol"):
        # Remove the "_vol" suffix and add "_nuclei"
        modified_string = input_string[:-4] + "_nuclei"
    else:
        # If the string does not end with "_vol", simply add "_nuclei"
        modified_string = input_string + "_nuclei"
    return  modified_string

def find_local_maxima(array):
    # Initial mask with all False
    max_mask = np.zeros_like(array, dtype=bool)
    
    # Iterate through each axis
    for axis in range(3):
        # Compare with the shifted version of itself along each axis
        greater_than_prev = array > np.roll(array, 1, axis=axis)
        greater_than_next = array > np.roll(array, -1, axis=axis)
        non_zero = array >0
        # Update the mask where a local maximum is found along the current axis
        max_mask |= greater_than_prev & greater_than_next & non_zero
    return max_mask

def skelet_test(im_file,surf_file,scales):
    vol_img=getImage(im_file)
    mesh=pv.read(surf_file)
    points_px=mesh.point_data['Coord px']

    from scipy import ndimage
    from scipy.ndimage import label
    from skimage import measure, io

    #transform=ndimage.distance_transform_edt(vol_img,sampling=scales)
    #np.save(r'C:\Users\kotzm\transform.npy',transform)
    transform=np.load(r'C:\Users\kotzm\transform.npy')
    maximas=find_local_maxima(transform)
    print('max')
    labeled_image = measure.label(maximas, connectivity=2)
    print('lab')
    properties = measure.regionprops(labeled_image)
    largest_component = max(properties, key=lambda prop: prop.area)
    print('area')
    largest_component_mask = labeled_image == largest_component.label


    viewer = napari.Viewer(ndisplay=3)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    surface = (points_px, faces)
    viewer.add_surface(surface)
    viewer.add_labels(vol_img)
    viewer.add_labels(maximas)
    viewer.add_labels(largest_component_mask)
    viewer.add_image(transform)
    #viewer.add_labels(blur)
    #viewer.add_points(points_plot)
    napari.run()


def make_Thickness():
    image_folder_list=os.listdir(vol_images_path)
    image_folder_list = [item for item in image_folder_list if os.path.isdir(os.path.join(vol_images_path, item))]
    image_folder_list=['20220611_mAG-zGem_H2a-mcherry_102hpf_LM_B3_analyzed_vol']
    for image_folder in image_folder_list:
        print(image_folder)
        
        image_dir_path=os.path.join(vol_images_path,image_folder)
        image_folder=get_name(image_folder)
        LMcoord_dir_path=os.path.join(LMcoord_path,image_folder+'_LMcoord')
 
        PastMetaData=evalStatus_Thickness(image_dir_path,LMcoord_dir_path)
        if not isinstance(PastMetaData,dict):
            continue

        MetaData_image=PastMetaData['vol_image_MetaData']
        writeJSON(LMcoord_dir_path,'vol_image_MetaData',MetaData_image)
        image_file=MetaData_image['vol image file name']

        MetaData_Coord=PastMetaData['Coord_MetaData']
        Surface_file_name=MetaData_Coord['Surface file']


        scales=MetaData_image['XYZ size in mum'].copy()
        if MetaData_image['axes']=='ZYX':
            scales[0], scales[-1] = scales[-1], scales[0]

        #actual calculation
        mesh = calculate_Thickness(os.path.join(image_dir_path,image_file),os.path.join(LMcoord_dir_path,Surface_file_name),scales)
        
        Surface_file=os.path.join(LMcoord_dir_path,Surface_file_name)
        mesh.save(Surface_file)

        MetaData_Thickness={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_Thickness['git hash']=sha
        MetaData_Thickness['git repo']='landmark_coordinate'
        MetaData_Thickness['Thickness version']=Thickness_version
        MetaData_Thickness['Surface file']=Surface_file_name
        MetaData_Thickness['XYZ size in mum']=MetaData_image['XYZ size in mum']
        MetaData_Thickness['axes']=MetaData_image['axes']
        MetaData_Thickness['experimentalist']=MetaData_image['experimentalist']
        MetaData_Thickness['genotype']=MetaData_image['genotype']
        MetaData_Thickness['is control']=MetaData_image['is control']
        MetaData_Thickness['time in hpf']=MetaData_image['time in hpf']
        MetaData_Thickness['input Surface checksum']=PastMetaData['Coord_MetaData']['output Surface checksum']
        check_Surface=get_checksum(Surface_file, algorithm="SHA1")
        MetaData_Thickness['output Surface checksum']=check_Surface
        writeJSON(LMcoord_dir_path,'Thickness_MetaData',MetaData_Thickness)


def test():   
    pass

if __name__ == "__main__":
    #test()
    make_Thickness() 

