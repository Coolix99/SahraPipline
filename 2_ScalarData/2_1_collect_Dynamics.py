from functools import reduce
import os
import sys
import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm
from scipy.spatial import cKDTree

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from IO import *


# ============================================================
# Geometry helpers (same as mesh-paper pipeline)
# ============================================================

def mesh_in_c1c2_space(mesh):
    c1 = mesh.point_data['coord_1']
    c2 = mesh.point_data['coord_2']
    pts = np.column_stack([c1, c2, np.zeros_like(c1)])
    mesh2d = mesh.copy()
    mesh2d.points = pts
    return mesh2d


def extract_contour_length(mesh2d, normal, origin):
    sliced = mesh2d.slice(normal=normal, origin=origin)
    if sliced.n_points < 2:
        return 0.0, None

    lines = sliced.split_bodies()
    max_len = 0.0
    best_line = None

    for line in lines:
        if line.n_points < 2:
            continue
        if line.length > max_len:
            max_len = line.length
            best_line = line

    return max_len, best_line


def extract_PD_line(mesh2d, c2_value):
    return extract_contour_length(mesh2d, normal=(0, 1, 0), origin=(0, c2_value, 0))


def extract_AP_line(mesh2d, c1_value):
    return extract_contour_length(mesh2d, normal=(1, 0, 0), origin=(c1_value, 0, 0))


# ============================================================
# Main
# ============================================================

def find_matching_subfolder(base_path: str, target: str):
    matches = [
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name)) and name.startswith(target)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def main():

    finmasks_folder_list = [
        item for item in os.listdir(finmasks_path)
        if os.path.isdir(os.path.join(finmasks_path, item))
    ]

    data_list = []

    for mask_folder in tqdm(finmasks_folder_list, desc="Processing folders", unit="folder"):
        mask_folder_path = os.path.join(finmasks_path, mask_folder)

        maskMetaData = get_JSON(mask_folder_path)
        if not maskMetaData:
            continue

        membrane_folder_path = os.path.join(membranes_path, mask_folder)
        MetaData_mem = get_JSON(membrane_folder_path)

        mask_img = getImage(
            os.path.join(mask_folder_path, maskMetaData['MetaData_finmasks']['finmasks file'])
        )

        if not MetaData_mem:
            MetaData_mem['MetaData_membrane'] = maskMetaData['MetaData_finmasks']

        MetaData_mem = MetaData_mem['MetaData_membrane']

        voxel_size = reduce(lambda x, y: x * y, MetaData_mem['scales ZYX'])
        Volume = np.sum(mask_img > 0) * voxel_size

        # ----------------------------------------------------
        # FlatFin mesh + metadata
        # ----------------------------------------------------
        FlatFin_dir = os.path.join(FlatFin_path, mask_folder + '_FlatFin')
        MetaData_flat = get_JSON(FlatFin_dir)

        if 'Thickness_MetaData' not in MetaData_flat:
            continue

        MetaData_flat = MetaData_flat['Thickness_MetaData']
        mesh = pv.read(os.path.join(FlatFin_dir, MetaData_flat['Surface file']))

        coord_1 = mesh.point_data['coord_1']
        coord_2 = mesh.point_data['coord_2']
        thickness = mesh.point_data['thickness']

        # ----------------------------------------------------
        # Volume / surface / integrated thickness
        # ----------------------------------------------------
        total_surface_area = mesh.area

        cell_areas = mesh.compute_cell_sizes()['Area']
        faces = mesh.faces.reshape((-1, 4))[:, 1:]

        total_integrated_thickness = 0.0
        for i, cell in enumerate(faces):
            avg_thickness = np.mean(thickness[cell])
            total_integrated_thickness += cell_areas[i] * avg_thickness

        # ----------------------------------------------------
        # NEW PD / AP geometry
        # ----------------------------------------------------
        mesh2d = mesh_in_c1c2_space(mesh)

        c1_min, c1_max = coord_1.min(), coord_1.max()
        c2_min, c2_max = coord_2.min(), coord_2.max()

        L_PD_BB = c1_max - c1_min
        L_AP_BB = c2_max - c2_min

        mid_AP = c2_min + 0.5 * L_AP_BB
        L_PD_midline, _ = extract_PD_line(mesh2d, mid_AP)

        PD_40 = c1_min + 0.4 * L_PD_BB
        L_AP_40line, _ = extract_AP_line(mesh2d, PD_40)

        PD_positions = np.linspace(c1_min, c1_max, 200)
        L_AP_longline = 0.0
        PD_long_rel = np.nan

        for PD in PD_positions:
            L, _ = extract_AP_line(mesh2d, PD)
            if L > L_AP_longline:
                L_AP_longline = L
                PD_long_rel = (PD - c1_min) / L_PD_BB

        # ----------------------------------------------------
        # NEW DV (thickness) via KD-tree
        # ----------------------------------------------------
        points_2d = np.column_stack([coord_1, coord_2])
        tree = cKDTree(points_2d)

        radius = 10.0
        idx = tree.query_ball_point([PD_40, mid_AP], r=radius)

        L_DV = float(np.mean(thickness[idx])) if idx else np.nan

        # ----------------------------------------------------
        # Base data
        # ----------------------------------------------------
        data = {
            'Mask Folder': mask_folder,
            'Volume': Volume,
            'Surface Area': total_surface_area,
            'Integrated Thickness': total_integrated_thickness,

            'L_PD_BB': L_PD_BB,
            'L_PD_midline': L_PD_midline,

            'L_AP_BB': L_AP_BB,
            'L_AP_40line': L_AP_40line,
            'L_AP_longline': L_AP_longline,
            'PD_long_rel': PD_long_rel,

            'L_DV': L_DV,
            'L_DV_npts': len(idx),

            'condition': MetaData_flat.get('condition'),
            'time in hpf': MetaData_flat.get('time in hpf'),
            'experimentalist': MetaData_flat.get('experimentalist'),
            'genotype': MetaData_flat.get('genotype'),
        }

        # ----------------------------------------------------
        # ED CELLS (BB ONLY)
        # ----------------------------------------------------
        subfolder_ED = find_matching_subfolder(ED_cells_path, mask_folder)
        if subfolder_ED is None:
            data_list.append(data)
            continue

        ED_cells_folder = os.path.join(ED_cells_path, subfolder_ED)
        MetaData_ED = get_JSON(ED_cells_folder)
        cell_file = MetaData_ED['MetaData_EDcells']['EDcells file']

        ED_cells_img = getImage(os.path.join(ED_cells_folder, cell_file))
        ED_mask = ED_cells_img > 0

        ED_mask_volume = np.sum(ED_mask) * voxel_size
        N_ED_cells = len(np.unique(ED_cells_img)) - (1 if 0 in ED_cells_img else 0)

        mask_voxels = np.column_stack(np.where(ED_mask))
        scales = np.array(MetaData_mem['scales ZYX'])
        mask_points_world = mask_voxels * scales

        tree = cKDTree(mesh.points)
        _, closest_idx = tree.query(mask_points_world)

        mesh_mask = np.zeros(mesh.n_points, dtype=bool)
        mesh_mask[np.unique(closest_idx)] = True

        coord1_ED = coord_1[mesh_mask]
        coord2_ED = coord_2[mesh_mask]

        L_PD_ED = coord1_ED.max() - coord1_ED.min()
        L_AP_ED = coord2_ED.max() - coord2_ED.min()

        area_ED = mesh.extract_points(mesh_mask).area

        total_integrated_thickness_ED = 0.0
        for i, cell in enumerate(faces):
            if np.all(mesh_mask[cell]):
                avg_thickness = np.mean(thickness[cell])
                total_integrated_thickness_ED += cell_areas[i] * avg_thickness

        data.update({
            'ED Mask Volume': ED_mask_volume,
            'N ED cells': N_ED_cells,
            'Surface Area ED': area_ED,
            'Integrated Thickness ED': total_integrated_thickness_ED,
            'L_PD_ED': L_PD_ED,
            'L_AP_ED': L_AP_ED,
        })

        data_list.append(data)
        
    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    df = pd.DataFrame(data_list)
    out_file = os.path.join(scalar_path, 'scalarGrowthData_meshBased.csv')
    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
