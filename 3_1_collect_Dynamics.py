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

        #DV axis
        coord_1 = mesh.point_data.get('coord_1', None)
        coord_2 = mesh.point_data.get('coord_2', None)
        thickness = mesh.point_data.get('thickness', None)
        
        if coord_1 is None or coord_2 is None or thickness is None:
            print(f"UnExpected")
            raise
        
        threshold=10
        # Filter points based on coord_1 within threshold
        mask = (coord_2 >= -threshold) & (coord_2 <= threshold)
        filtered_data = {
            'coord_1': coord_1[mask],
            'thickness': thickness[mask],
        }

        # Sort by coord_2
        sorted_indices = filtered_data['coord_1'].argsort()
        c1 = filtered_data['coord_1'][sorted_indices]
        d = filtered_data['thickness'][sorted_indices]
        
        # Normalize coord_1
        c1 = c1 / np.max(c1)
        rel_pos=0.4
        # Apply the mask to filter values
        mask = (c1 >= rel_pos - 0.05) & (c1 <= rel_pos + 0.05)
        filtered_d = d[mask]

        # Calculate mean or assign NaN
        if filtered_d.size > 0:
            DV=np.mean(filtered_d)
        else:
            raise

        MetaData=MetaData['Thickness_MetaData']
        data = {
            'Mask Folder': mask_folder,
            'Volume': Volume,
            'Surface Area': total_surface_area,
            'Integrated Thickness': total_integrated_thickness,
            'L PD': L_1,
            'L AP': L_2,
            'L DV': DV,
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
    df.to_csv(os.path.join(Curv_Thick_path,'scalarGrowthData.csv'), index=False,sep=';')
        

if __name__ == "__main__":
    main()