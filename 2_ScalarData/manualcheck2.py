import pyvista as pv
import numpy as np
import napari
import os
import pandas as pd

main_folder = "/media/grp07_max/sahra_shivani_data/sorted_data/for_curv_thick/FlatFin"
sample = "20240103sox10_clau126hpfreg4_Stitch"
# sample = '20240103sox10_clau126hpfreg3_Stitch'
vtk_path = os.path.join(main_folder, sample+'_FlatFin', sample + "_surface.vtk")
vtk_path = "/home/max/Downloads/20240103sox10_clau126hpfreg4_Stitch_FlatFin_sox.vtk"
mesh = pv.read(vtk_path)

faces = mesh.faces.reshape((-1, 4))[:, 1:]

coords = np.column_stack((mesh.point_data["coord_1"],
                          mesh.point_data["coord_2"]))

# pick a field that exists
values = mesh.point_data["thickness"]

# Excel path
excel_path = main_folder + "/excel/" + sample + "_point_measurements.xlsx"

# Empty DF
df = pd.DataFrame(columns=["name", "p1_x", "p1_y", "p2_x", "p2_y"])

# --- Napari viewer ---
viewer = napari.Viewer()
viewer.add_surface((coords, faces, values), name="Mesh", colormap="turbo")

points_layer = viewer.add_points(name="Selected Points")

def on_points_change(event):
    data = points_layer.data
    if len(data) == 2:
        p1, p2 = data
        df.loc[0] = [sample, p1[0], p1[1], p2[0], p2[1]]

        # Ensure output folder exists
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)

        # Save Excel
        df.to_excel(excel_path, index=False)
        print(f"Saved 2 points → {excel_path}")

points_layer.events.data.connect(on_points_change)

napari.run()