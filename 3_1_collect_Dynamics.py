from functools import reduce
import pyvista as pv
import pandas as pd

from config import *
from IO import *

def main():
    finmasks_folder_list= [item for item in os.listdir(finmasks_path) if os.path.isdir(os.path.join(finmasks_path, item))]
    data_list = []
    for mask_folder in finmasks_folder_list:
        print(mask_folder)
        mask_folder_path=os.path.join(finmasks_path,mask_folder)
        
        maskMetaData=get_JSON(mask_folder_path)
        if maskMetaData=={}:
            print('no mask')
            continue
        membrane_folder_path=os.path.join(membranes_path,mask_folder)
        MetaData=get_JSON(membrane_folder_path)

        mask_img=getImage(os.path.join(mask_folder_path,maskMetaData['MetaData_finmasks']['finmasks file']))
        if  MetaData=={}:
            MetaData['MetaData_membrane']=maskMetaData['MetaData_finmasks']
        MetaData=MetaData['MetaData_membrane']
        voxel_size=reduce(lambda x, y: x * y, MetaData['scales ZYX'])
        Volume=np.sum(mask_img>0)*voxel_size

        FlatFin_dir_path = os.path.join(FlatFin_path, mask_folder+'_FlatFin')
        MetaData = get_JSON(FlatFin_dir_path)
        if 'Thickness_MetaData' not in MetaData:
            continue
        
        mesh=pv.read(os.path.join(FlatFin_dir_path, MetaData['Thickness_MetaData']['Surface file']))
        L_1=np.max(mesh.point_data['coord_1'])-np.min(mesh.point_data['coord_1'])
        L_2=np.max(mesh.point_data['coord_2'])-np.min(mesh.point_data['coord_2'])
        
        total_surface_area = mesh.area
        thickness_data = mesh.point_data['thickness']

        cell_areas = mesh.compute_cell_sizes()['Area']
        total_integrated_thickness = 0.0
        for i, cell in enumerate(mesh.faces.reshape((-1, 4))[:, 1:]):  # cells are stored as (N, 4) where first number is # of points
            # Get the points of the current cell (triangle)
            points_ids = cell
            points_thickness = thickness_data[points_ids]
            avg_thickness = np.mean(points_thickness)
            total_integrated_thickness += cell_areas[i] * avg_thickness

        MetaData=MetaData['Thickness_MetaData']
        data = {
            'Mask Folder': mask_folder,
            'Volume': Volume,
            'Surface Area': total_surface_area,
            'Integrated Thickness': total_integrated_thickness,
            'L PD': L_1,
            'L AP': L_2,
            'condition': MetaData.get('condition', None),
            'time in hpf': MetaData.get('time in hpf', None),
            'experimentalist': MetaData.get('experimentalist', None),
            'genotype': MetaData.get('genotype', None)
        }
        
        data_list.append(data)
        
    
    # After loop, convert list of dictionaries to DataFrame
    df = pd.DataFrame(data_list)
    print(df)
    df.to_hdf(os.path.join(Curv_Thick_path,'scalarGrowthData.h5'), key='data', mode='w')
        

if __name__ == "__main__":
    main()