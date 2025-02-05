import os
import shutil
import pyvista as pv
from zf_pf_geometry.metadata_manager import get_JSON

def plot_all_surfaces(surface_dir, state="latest", show_latest_only=True):
    """
    Processes and plots surfaces in the given directory based on their state and metadata.

    Args:
        surface_dir (str): Path to the directory containing surface data.
        state (str): Desired state of the surface ('surface', 'coord', or 'thickness').
        show_latest_only (bool): Whether to show only the latest state.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    surface_folder_list = [
        item for item in os.listdir(surface_dir) if os.path.isdir(os.path.join(surface_dir, item))
    ]
    logger.info(f"Found {len(surface_folder_list)} surface folders for processing.")

    for data_name in surface_folder_list:
        folder_path = os.path.join(surface_dir, data_name)

        # Load metadata
        metadata = get_JSON(folder_path)
        if not metadata or state not in metadata:
            logger.warning(f"Skipping {data_name}: Missing metadata or state '{state}' not found.")
            continue

        # Load surface file
        surface_file_key = f"{state.capitalize()} file name"
        if surface_file_key not in metadata[state]:
            logger.warning(f"Skipping {data_name}: Surface file key '{surface_file_key}' missing in metadata.")
            continue

        surface_file_name = metadata[state][surface_file_key]
        surface_file_path = os.path.join(folder_path, surface_file_name)

        if not os.path.exists(surface_file_path):
            logger.warning(f"Skipping {data_name}: Surface file '{surface_file_path}' not found.")
            continue

        logger.info(f"Plotting surface for {data_name} from file: {surface_file_path}")

        # Load the surface mesh
        mesh = pv.read(surface_file_path)
        logger.info(f"Loaded surface mesh with {mesh.n_points} points and {mesh.n_cells} cells.")

        # Create the plotter
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars='thickness' if 'thickness' in mesh.point_data else None)

        # Define a callback function for interactive deletion
        def key_callback(key):
            logger.info(f"Key pressed: {key}")
            if key in {'d', 'r'}:
                logger.info(f"Deleting surface: {surface_file_path}")
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    logger.error(f"Failed to delete {folder_path}: {e}")
            if key == 'r':
                logger.info(f"Additional cleanup for {data_name}.")
                # Implement additional cleanup logic if needed
            plotter.close()
            return False

        # Bind the key callbacks
        plotter.add_key_event('d', lambda: key_callback('d'))  # Delete only the surface
        plotter.add_key_event('r', lambda: key_callback('r'))  # Delete surface and associated data
        plotter.add_key_event('q', plotter.close)             # Quit without deleting

        # Show the plot
        try:
            plotter.show()
        except Exception as e:
            logger.error(f"Error displaying plot for {data_name}: {e}")
