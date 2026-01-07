import pyvista as pv
import numpy as np
import napari
import os

# --- Load your mesh ---
main_folder = "/media/grp07_max/sahra_shivani_data/sorted_data/for_curv_thick/FlatFin"
sample = "20240103sox10_clau126hpfreg4_Stitch"
vtk_path = os.path.join(main_folder, sample+'_FlatFin', sample + "_surface.vtk")
mesh = pv.read(vtk_path)

# Extract triangle connectivity (faces)
# PyVista stores faces as [3, i, j, k, 3, i, j, k, ...]
faces = mesh.faces.reshape((-1, 4))[:, 1:]   # drop the leading "3" column

# Extract vertex coordinates in the 2D flattened coordinate system
coords = np.column_stack((mesh.point_data["coord_1"],
                          mesh.point_data["coord_2"]))

# Thickness values at vertices
thickness = mesh.point_data["thickness"]

# --- Display in napari ---
viewer = napari.Viewer()

viewer.add_surface(
    data=(coords, faces, thickness),
    name="Triangulated Thickness Mesh",
    colormap="turbo"
)

napari.run()