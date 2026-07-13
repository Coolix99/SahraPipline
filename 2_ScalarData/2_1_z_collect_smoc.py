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
    return extract_contour_length(
        mesh2d,
        normal=(0, 1, 0),
        origin=(0, c2_value, 0),
    )


def extract_AP_line(mesh2d, c1_value):
    return extract_contour_length(
        mesh2d,
        normal=(1, 0, 0),
        origin=(c1_value, 0, 0),
    )


def get_membrane_metadata(mask_metadata, membrane_folder_path):
    membrane_metadata = get_JSON(membrane_folder_path)

    if membrane_metadata:
        return membrane_metadata['MetaData_membrane']

    return mask_metadata['MetaData_finmasks']


def get_voxel_scales(membrane_metadata, mask_metadata):
    try:
        return membrane_metadata['scales ZYX']
    except KeyError:
        return mask_metadata['metaData']['scales ZYX']


def calculate_integrated_thickness(mesh, thickness):
    cell_areas = mesh.compute_cell_sizes()['Area']
    faces = mesh.faces.reshape((-1, 4))[:, 1:]

    total_integrated_thickness = 0.0

    for i, cell in enumerate(faces):
        avg_thickness = np.mean(thickness[cell])
        total_integrated_thickness += cell_areas[i] * avg_thickness

    return total_integrated_thickness


def calculate_pd_ap_geometry(mesh, coord_1, coord_2):
    mesh2d = mesh_in_c1c2_space(mesh)

    c1_min = coord_1.min()
    c1_max = coord_1.max()
    c2_min = coord_2.min()
    c2_max = coord_2.max()

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
        length, _ = extract_AP_line(mesh2d, PD)

        if length > L_AP_longline:
            L_AP_longline = length
            PD_long_rel = (PD - c1_min) / L_PD_BB

    return {
        'L_PD_BB': L_PD_BB,
        'L_PD_midline': L_PD_midline,
        'L_AP_BB': L_AP_BB,
        'L_AP_40line': L_AP_40line,
        'L_AP_longline': L_AP_longline,
        'PD_long_rel': PD_long_rel,
        'PD_40': PD_40,
        'mid_AP': mid_AP,
    }


def calculate_dv_thickness(coord_1, coord_2, thickness, PD_40, mid_AP):
    points_2d = np.column_stack([coord_1, coord_2])
    tree = cKDTree(points_2d)

    radius = 10.0
    idx = tree.query_ball_point([PD_40, mid_AP], r=radius)

    L_DV = float(np.mean(thickness[idx])) if idx else np.nan

    return L_DV, len(idx)


def process_mask_folder(mask_folder):
    mask_folder_path = os.path.join(finmasks_path, mask_folder)
    mask_metadata = get_JSON(mask_folder_path)

    if not mask_metadata:
        return None

    FlatFin_dir = os.path.join(FlatFin_path, mask_folder + '_FlatFin')
    flatfin_metadata = get_JSON(FlatFin_dir)

    if not flatfin_metadata:
        return None

    if 'Thickness_MetaData' not in flatfin_metadata:
        return None

    thickness_metadata = flatfin_metadata['Thickness_MetaData']

    if thickness_metadata.get('genotype') != 'Smoc12':
        return None

    membrane_folder_path = os.path.join(membranes_path, mask_folder)
    membrane_metadata = get_membrane_metadata(
        mask_metadata,
        membrane_folder_path,
    )

    mask_file = mask_metadata['MetaData_finmasks']['finmasks file']
    mask_img = getImage(os.path.join(mask_folder_path, mask_file))

    zyx = get_voxel_scales(membrane_metadata, mask_metadata)
    voxel_size = reduce(lambda x, y: x * y, zyx)
    volume = np.sum(mask_img > 0) * voxel_size

    surface_file = thickness_metadata['Surface file']
    mesh = pv.read(os.path.join(FlatFin_dir, surface_file))

    coord_1 = mesh.point_data['coord_1']
    coord_2 = mesh.point_data['coord_2']
    thickness = mesh.point_data['thickness']

    total_surface_area = mesh.area
    total_integrated_thickness = calculate_integrated_thickness(
        mesh,
        thickness,
    )

    geometry = calculate_pd_ap_geometry(
        mesh,
        coord_1,
        coord_2,
    )

    L_DV, L_DV_npts = calculate_dv_thickness(
        coord_1,
        coord_2,
        thickness,
        geometry['PD_40'],
        geometry['mid_AP'],
    )

    return {
        'Mask Folder': mask_folder,
        'Volume': volume,
        'Surface Area': total_surface_area,
        'Integrated Thickness': total_integrated_thickness,
        'L_PD_BB': geometry['L_PD_BB'],
        'L_PD_midline': geometry['L_PD_midline'],
        'L_AP_BB': geometry['L_AP_BB'],
        'L_AP_40line': geometry['L_AP_40line'],
        'L_AP_longline': geometry['L_AP_longline'],
        'PD_long_rel': geometry['PD_long_rel'],
        'L_DV': L_DV,
        'L_DV_npts': L_DV_npts,
        'condition': thickness_metadata.get('condition'),
        'time in hpf': thickness_metadata.get('time in hpf'),
        'experimentalist': thickness_metadata.get('experimentalist'),
        'genotype': thickness_metadata.get('genotype'),
    }


def main():
    finmasks_folder_list = [
        item
        for item in os.listdir(finmasks_path)
        if os.path.isdir(os.path.join(finmasks_path, item))
    ]

    data_list = []

    for mask_folder in tqdm(
        finmasks_folder_list,
        desc="Processing Smoc12 folders",
        unit="folder",
    ):
        data = process_mask_folder(mask_folder)

        if data is not None:
            data_list.append(data)

    df = pd.DataFrame(data_list)

    out_file = os.path.join(
        scalar_path,
        'scalarGrowthData_meshBased_Smoc12.csv',
    )

    df.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")
    print(f"Number of Smoc12 samples: {len(df)}")


if __name__ == "__main__":
    main()