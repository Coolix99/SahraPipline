import os
import numpy as np
import pyvista as pv
import logging
from scipy.spatial import cKDTree
from tqdm import tqdm
from zf_pf_geometry.utils import load_tif_image
from simple_file_checksum import get_checksum
from zf_pf_geometry.metadata_manager import should_process, write_JSON


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



def process_geometry(geometry_dir, output_dir):
    """
    Processes all geometry files by projecting images onto surfaces.

    Args:
        geometry_dir (str): Path to the geometry folder containing subfolders with `.vtk` files.
        output_dir (str): Path to save the updated surfaces.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all subdirectories in geometry_dir
    geometry_subfolders = [d for d in os.listdir(geometry_dir) if os.path.isdir(os.path.join(geometry_dir, d))]

    for dataset_name in tqdm(geometry_subfolders, desc="Processing Surfaces", unit="dataset"):
        dataset_path = os.path.join(geometry_dir, dataset_name)

        # Define output path
        output_path = os.path.join(output_dir, dataset_name)
        os.makedirs(output_path, exist_ok=True)

        # Check if processing is needed
        bre_folder_path=os.path.join(bre_dir,dataset_name)
        smoc_folder_path=os.path.join(smoc_dir,dataset_name)
        res = should_process([dataset_path, bre_folder_path, smoc_folder_path], ['thickness', 'Image_BRE', 'Image_Smoc'], output_path, "projected_data")
       
        if not res:
            logger.info(f"Skipping {dataset_name}: No processing needed.")
            continue

        input_data, input_checksum = res
        scale=np.array(input_data['thickness']['scale'])

        # Load surface
        surface = pv.read(os.path.join(dataset_path,input_data['thickness']['Surface(Thickness) file name']))
       
        # Load BRE and Smoc images
        try:
            bre_image = load_tif_image(bre_folder_path)
            logger.info(f"Loaded BRE image for {dataset_name}.")
        except Exception as e:
            logger.error(f"Failed to load BRE image for {dataset_name}: {e}")
            continue

        try:
            smoc_image = load_tif_image(smoc_folder_path)
            logger.info(f"Loaded Smoc image for {dataset_name}.")
        except Exception as e:
            logger.error(f"Failed to load Smoc image for {dataset_name}: {e}")
            continue

        # Project image data onto surface
        logger.info(f"Projecting BRE data for {dataset_name}.")
        surface = project_image_to_surface(surface, bre_image,scale,'BRE')

        logger.info(f"Projecting Smoc data for {dataset_name}.")
        surface = project_image_to_surface(surface, smoc_image,scale,'Smoc')


        surface.save(os.path.join(output_path, dataset_name+ ".vtk"))
        logger.info(f"Saved updated surface: {output_path}")

        # Update metadata
        updated_metadata = input_data["thickness"]
        updated_metadata["Projected Surface file name"] = dataset_name + ".vtk"
        updated_metadata['input_data_checksum'] = input_checksum
        updated_metadata["Projected Surface checksum"] = get_checksum(output_path, algorithm="SHA1")

        write_JSON(output_path, "projected_data", updated_metadata)

    logger.info("Projection processing completed.")

if __name__ == "__main__":
    # Define folder paths
    geometry_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/for_curv_thick/FlatFin"
    output_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/projected_surfaces"

    bre_dir = "/home/max/Documents/02_Data/structured_data/structured_vinita/BRE"
    smoc_dir = "/home/max/Documents/02_Data/structured_data/structured_vinita/Smoc"
    # Run processing
    process_geometry(geometry_dir, bre_dir, smoc_dir, output_dir)
