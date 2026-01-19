import os
import numpy as np
import pyvista as pv
import logging
from scipy.spatial import cKDTree
from tqdm import tqdm
import pandas as pd
from zf_pf_geometry.utils import load_tif_image
from simple_file_checksum import get_checksum
from zf_pf_geometry.metadata_manager import should_process, write_JSON
from zf_pf_diffeo.project import project_df_surface

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



def process_geometry(geometry_dir, cell_props_dir, output_dir):
    """
    Processes all geometry files by projecting images onto surfaces.

    Args:
        geometry_dir (str): Path to the geometry folder containing subfolders with `.vtk` files.
        output_dir (str): Path to save the updated surfaces.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all subdirectories in geometry_dir
    geometry_subfolders = [d for d in os.listdir(geometry_dir) if os.path.isdir(os.path.join(geometry_dir, d))]

    for geometry_name in tqdm(geometry_subfolders, desc="Processing Surfaces", unit="dataset"):
        dataset_name = geometry_name.removesuffix("_FlatFin")
        dataset_path = os.path.join(geometry_dir, geometry_name)

        # Define output path
        output_path = os.path.join(output_dir, dataset_name)
        os.makedirs(output_path, exist_ok=True)

        # Check if processing is needed
        cell_props_folder_path=os.path.join(cell_props_dir,dataset_name)
       
        res = should_process([dataset_path, cell_props_folder_path], ['Thickness_MetaData', 'MetaData_EDcell_props'], output_path, "projected_data")
        
        if not res:
            logger.info(f"Skipping {dataset_name}: No processing needed.")
            continue

        input_data, input_checksum = res

        #scale=np.array(input_data['Thickness_MetaData']['scales ZYX'])
   
        # Load surface
        surface = pv.read(os.path.join(dataset_path,input_data['Thickness_MetaData']['Surface file']))
       
        # Load cellprop df
        cell_props_file=os.path.join(cell_props_folder_path,input_data["MetaData_EDcell_props"]["EDcell_props file"])
        cell_prop_df=pd.read_csv(cell_props_file,sep=';')
        
        cell_prop_df['2I1/I2+I3']=2*cell_prop_df['moments_eigvals 1']/(cell_prop_df['moments_eigvals 2']+cell_prop_df['moments_eigvals 3'])
        #print(cell_prop_df.head())

        # Project image data onto surface
        logger.info(f"Projecting cell props {dataset_name}.")
        surface = project_df_surface(surface, cell_prop_df,['centroids_scaled Z',  'centroids_scaled Y',  'centroids_scaled X'],
                                     ['Volume','Surface Area','Solidity','2I1/I2+I3'])
        #print(surface.point_data)
        
        output_file=os.path.join(output_path, dataset_name+ ".vtk")
        surface.save(output_file)
        logger.info(f"Saved updated surface: {output_path}")

        # Update metadata
        updated_metadata = input_data["Thickness_MetaData"]
        updated_metadata["Projected Surface file name"] = dataset_name + ".vtk"
        updated_metadata['input_data_checksum'] = input_checksum
        updated_metadata["Projected Surface checksum"] = get_checksum(output_file, algorithm="SHA1")

        write_JSON(output_path, "projected_data", updated_metadata)

    logger.info("Projection processing completed.")

if __name__ == "__main__":
    # Define folder paths
    geometry_dir = FlatFin_path
    output_dir = os.path.join(Output_path,"morphoMaps","projected_surfaces")
    # Run processing
    process_geometry(geometry_dir, ED_cell_props_path, output_dir)
