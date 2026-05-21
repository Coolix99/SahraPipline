import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from IO import *

csv_file = os.path.join(scalar_path, 'scalarGrowthData_meshBased.csv')
df = pd.read_csv(csv_file)

df_nuclei = pd.read_excel(
    os.path.join('/home/max/Downloads', 'WT_regenartion_newest.xlsx')
)

# -----------------------------------------------------------------------------
# Filter for:
#   genotype   = WT
#   condition  = Regeneration
#   time in hpf = 84
# -----------------------------------------------------------------------------
TARGET_HPF = 84

# --- mesh-based dataframe ---
df_mesh = df[
    (df['genotype'] == 'WT') &
    (df['condition'] == 'Regeneration') &
    (df['time in hpf'] == TARGET_HPF)
].copy()

# --- Nuclei dataframe ---
# Column names are already the same as shown in your print output:
# genotype, condition, time in hpf, experimentalist
df_excel = df_nuclei[
    (df_nuclei['genotype'] == 'WT') &
    (df_nuclei['condition'] == 'Regeneration') &
    (df_nuclei['time in hpf'] == TARGET_HPF)
].copy()

# -----------------------------------------------------------------------------
# Compute derived quantity: Volume / Surface Area
# -----------------------------------------------------------------------------
df_mesh['L_DV'] = df_mesh['Volume'] / df_mesh['Surface Area']
df_excel['L_DV'] = (
    df_excel['Volume'] / df_excel['Surface Area']
)

# -----------------------------------------------------------------------------
# Print summary statistics
# -----------------------------------------------------------------------------
def print_summary(name, d):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"N = {len(d)}")

    for col in ['Volume', 'Surface Area', 'L_DV']:
        if len(d) > 0:
            print(
                f"{col:20s}: "
                f"mean = {d[col].mean():.3f}, "
                f"std = {d[col].std():.3f}, "
                f"min = {d[col].min():.3f}, "
                f"max = {d[col].max():.3f}"
            )
        else:
            print(f"{col:20s}: no data")

print_summary("Membrane", df_mesh)
print_summary("Nuclei file", df_excel)

# -----------------------------------------------------------------------------
# Prepare plotting
# -----------------------------------------------------------------------------
metrics = ['Volume', 'Surface Area', 'L_DV']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f'WT Regeneration at {TARGET_HPF} hpf\n'
    'Comparison: Membrane vs Nuclei',
    fontsize=14
)

for ax, metric in zip(axes, metrics):

    # Scatter all individual points with small x-offsets
    x_mesh = np.zeros(len(df_mesh)) + np.random.uniform(-0.05, 0.05, len(df_mesh))
    x_excel = np.ones(len(df_excel)) + np.random.uniform(-0.05, 0.05, len(df_excel))

    ax.scatter(
        x_mesh,
        df_mesh[metric],
        s=80,
        alpha=0.8,
        label='Membrane',
        marker='o'
    )

    ax.scatter(
        x_excel,
        df_excel[metric],
        s=80,
        alpha=0.8,
        label='Nuclei',
        marker='s'
    )

    # Plot means and standard deviations
    if len(df_mesh) > 0:
        ax.errorbar(
            0,
            df_mesh[metric].mean(),
            yerr=df_mesh[metric].std(),
            fmt='k_',
            linewidth=3,
            capsize=6
        )

    if len(df_excel) > 0:
        ax.errorbar(
            1,
            df_excel[metric].mean(),
            yerr=df_excel[metric].std(),
            fmt='k_',
            linewidth=3,
            capsize=6
        )

    # Formatting
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Membrane', 'Nuclei'])
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)

# Show legend only once
axes[0].legend()

plt.tight_layout()
plt.show()

if len(df_mesh) > 0:
    # Sort by L_DV in descending order and take the top 5
    top5 = (
        df_mesh
        .sort_values('L_DV', ascending=False)
        .head(5)
    )

    print("\nTop 5 membrane samples with highest L_DV")
    print("=======================================================")

    for rank, (_, row) in enumerate(top5.iterrows(), start=1):
        print(f"\n#{rank}")
        print("-" * 40)

        if 'Mask Folder' in row.index:
            print(f"Mask Folder          : {row['Mask Folder']}")
        if 'experimentalist' in row.index:
            print(f"Experimentalist      : {row['experimentalist']}")
        if 'genotype' in row.index:
            print(f"Genotype             : {row['genotype']}")
        if 'condition' in row.index:
            print(f"Condition            : {row['condition']}")
        if 'time in hpf' in row.index:
            print(f"Time in hpf          : {row['time in hpf']}")

        print(f"Volume               : {row['Volume']:.6f}")
        print(f"Surface Area         : {row['Surface Area']:.6f}")
        print(f"L_DV  : {row['L_DV']:.6f}")

    # Optional: compact table view
    print("\nCompact summary:")
    cols = [
        col for col in [
            'Mask Folder',
            'experimentalist',
            'Volume',
            'Surface Area',
            'L_DV'
        ] if col in top5.columns
    ]
    print(top5[cols].to_string(index=False))

else:
    print("\nNo membrane data found for the selected condition.")


if len(df_excel) > 0:
    # Sort by L_DV in ascending order and take the top 5 smallest
    bottom5 = (
        df_excel
        .sort_values('L_DV', ascending=True)
        .head(5)
    )

    print("\nTop 5 nuclei samples with smallest L_DV")
    print("======================================================")

    for rank, (_, row) in enumerate(bottom5.iterrows(), start=1):
        print(f"\n#{rank}")
        print("-" * 40)

        if 'Mask Folder' in row.index:
            print(f"Mask Folder          : {row['Mask Folder']}")
        elif 'Name' in row.index:
            print(f"Name                 : {row['Name']}")

        if 'experimentalist' in row.index:
            print(f"Experimentalist      : {row['experimentalist']}")
        if 'genotype' in row.index:
            print(f"Genotype             : {row['genotype']}")
        if 'condition' in row.index:
            print(f"Condition            : {row['condition']}")
        if 'time in hpf' in row.index:
            print(f"Time in hpf          : {row['time in hpf']}")

        print(f"Volume               : {row['Volume']:.6f}")
        print(f"Surface Area         : {row['Surface Area']:.6f}")
        print(f"L_DV  : {row['L_DV']:.6f}")

    # Optional compact summary
    print("\nCompact summary:")
    cols = [
        col for col in [
            'Name',
            'Mask Folder',
            'experimentalist',
            'Volume',
            'Surface Area',
            'L_DV'
        ] if col in bottom5.columns
    ]
    print(bottom5[cols].to_string(index=False))

else:
    print("\nNo nuclei data found for the selected condition.")

import pyvista as pv
import napari

mesh=pv.read('/media/grp07_max/sahra_shivani_data/sorted_data/for_curv_thick/FlatFin/260425_reg_sox10_claudin-gfp_84h_pecfin4_FlatFin/260425_reg_sox10_claudin-gfp_84h_pecfin4_surface.vtk')
vol_img=getImage('/media/grp07_max/sahra_shivani_data/sorted_data/finmasks/260425_reg_sox10_claudin-gfp_84h_pecfin4/260425_reg_sox10_claudin-gfp_84h_pecfin4.tif').astype(int)
viewer = napari.Viewer(ndisplay=3)
faces = mesh.faces.reshape(-1, 4)[:, 1:]
points_px=mesh.point_data['Coord px']
surface = (points_px, faces)
viewer.add_surface(surface)
viewer.add_labels(vol_img)
napari.run()

mesh=pv.read('/media/grp07_max/sahra_shivani_data/sorted_data/for_curv_thick/FlatFin/060724_reg_sox10_claudin-gfp_84h_pecfin4_FlatFin/060724_reg_sox10_claudin-gfp_84h_pecfin4_surface.vtk')
vol_img=getImage('/media/grp07_max/sahra_shivani_data/sorted_data/finmasks/060724_reg_sox10_claudin-gfp_84h_pecfin4/060724_reg_sox10_claudin-gfp_84h_pecfin4.tif').astype(int)
viewer = napari.Viewer(ndisplay=3)
faces = mesh.faces.reshape(-1, 4)[:, 1:]
points_px=mesh.point_data['Coord px']
surface = (points_px, faces)
viewer.add_surface(surface)
viewer.add_labels(vol_img)
napari.run()


mesh=pv.read('/media/grp07_max/structured_data/images/fin_geometry/LM_coord/20231004_BRE-laux_GFP_H2A-mCh_84hpf_LM_1-1_nuclei_LMcoord/20231004_BRE-laux_GFP_H2A-mCh_84hpf_LM_1-1_nuclei_surface.vtk')
vol_img=getImage('/media/grp07_max/structured_data/images/vol_and_nuclei_mask/raw_images_vol/20231004_BRE-laux_GFP_H2A-mCh_84hpf_LM_1-1_vol/20231004_BRE-laux_GFP_H2A-mCh_84hpf_LM_1-1_vol.tif').astype(int)
viewer = napari.Viewer(ndisplay=3)
faces = mesh.faces.reshape(-1, 4)[:, 1:]
points_px=mesh.point_data['Coord px']
surface = (points_px, faces)
viewer.add_surface(surface)
viewer.add_labels(vol_img)
napari.run()