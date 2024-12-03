import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt

from config import *
from IO import *


def main():
    EDprops_folder_list= [item for item in os.listdir(ED_cell_props_path) if os.path.isdir(os.path.join(ED_cell_props_path, item))]
    for EDprop_folder in EDprops_folder_list:
        print(EDprop_folder)
        EDprop_folder_path=os.path.join(ED_cell_props_path,EDprop_folder)
        
        EDpropMetaData=get_JSON(EDprop_folder_path)
        if EDpropMetaData=={}:
            print('no EDprops')
            continue
        df_prop = pd.read_hdf(os.path.join(EDprop_folder_path,EDpropMetaData['MetaData_EDcell_props']['EDcells file']), key='data')
        df_proj = pd.read_hdf(os.path.join(EDprop_folder_path,EDpropMetaData['MetaData_EDcell_proj']['EDcell_proj file']), key='data')

        folder_path=os.path.join(FlatFin_path,EDprop_folder+'_FlatFin')
        FF_MetaData=get_JSON(folder_path)
        if FF_MetaData=={}:
            print('no FF_MetaData')
            continue
        surface_file_name=FF_MetaData['Thickness_MetaData']['Surface file']
        mesh=pv.read(os.path.join(folder_path,surface_file_name))

 
        print(df_proj)
        print(df_prop)
        print(mesh.point_data)

        merged_df = pd.merge(df_proj, df_prop, on='Label', suffixes=('_proj', '_prop'))

        # Compute vector v
        v = np.column_stack([
            merged_df['centroids_scaled Z'] - merged_df['Closest  Point Z'],
            merged_df['centroids_scaled Y'] - merged_df['Closest  Point Y'],
            merged_df['centroids_scaled X'] - merged_df['Closest  Point X']
        ])
       
        # Retrieve normal vectors
        normal_vectors = np.array([
            mesh.point_normals[index]
            for index in merged_df['Closest Mesh Point Index']
        ])

        # Compute scalar products
        scalar_products = np.einsum('ij,ij->i', v, normal_vectors)

        # Create histogram
        plt.hist(scalar_products, bins=30, alpha=0.7)
        plt.title('Histogram of Scalar Products')
        plt.xlabel('Scalar Product')
        plt.ylabel('Frequency')
        plt.show()

        # Calculate and print mean
        mean_scalar_product = np.mean(scalar_products)
        print(f'Mean Scalar Product: {mean_scalar_product}')
        return
        #todo:collect ofer all data

if __name__ == "__main__":
    main()