import os
import pandas as pd
from plotBokehHelper import plot_double_timeseries
from plotHelper.plotBokehHelper_old import plot_scatter_corner
from bokeh.io import show
import pyvista as pv

from config import *
from IO import *

def collect_dfs():
     # Initialize an empty list to collect all dataframes
    all_dfs = []

    # Iterate through folders
    EDprops_folder_list = [item for item in os.listdir(ED_cell_props_path) if os.path.isdir(os.path.join(ED_cell_props_path, item))]
    for EDprop_folder in EDprops_folder_list:
        print(EDprop_folder)
        EDprop_folder_path = os.path.join(ED_cell_props_path, EDprop_folder)

        EDpropMetaData = get_JSON(EDprop_folder_path)
        if not EDpropMetaData:
            print('No EDprops found')
            continue

        # Load the dataframe
        df_prop = pd.read_hdf(os.path.join(EDprop_folder_path, EDpropMetaData['MetaData_EDcell_props']['EDcell_props file']), key='data')
        
        # Add metadata as new columns
        df_prop['time in hpf'] = EDpropMetaData['MetaData_EDcell_props']['time in hpf']
        df_prop['condition'] = EDpropMetaData['MetaData_EDcell_props']['condition']
        df_prop['folder_name'] = EDprop_folder
        # Append the dataframe to the list
        all_dfs.append(df_prop)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

def collect_dfs_proj():
     # Initialize an empty list to collect all dataframes
    all_dfs = []

    # Iterate through folders
    EDprops_folder_list = [item for item in os.listdir(ED_cell_props_path) if os.path.isdir(os.path.join(ED_cell_props_path, item))]
    for EDprop_folder in EDprops_folder_list:
        print(EDprop_folder)
        EDprop_folder_path = os.path.join(ED_cell_props_path, EDprop_folder)

        EDpropMetaData = get_JSON(EDprop_folder_path)
        if not EDpropMetaData:
            print('No EDprops found')
            continue

        # Load the dataframe
        df_proj = pd.read_hdf(os.path.join(EDprop_folder_path, EDpropMetaData['MetaData_EDcell_proj']['EDcell_proj file']), key='data')
        if len(df_proj.columns)<9:
            print('projection incorrect!')
            continue
        # Add metadata as new columns
        df_proj['time in hpf'] = EDpropMetaData['MetaData_EDcell_props']['time in hpf']
        df_proj['condition'] = EDpropMetaData['MetaData_EDcell_props']['condition']

        folder_path=os.path.join(FlatFin_path,EDprop_folder+'_FlatFin')
        FF_MetaData=get_JSON(folder_path)
        if FF_MetaData=={}:
            print('no FF_MetaData')
            continue
        surface_file_name=FF_MetaData['Thickness_MetaData']['Surface file']
        mesh=pv.read(os.path.join(folder_path,surface_file_name))
        
        normal_vectors = np.array([
            mesh.point_normals[index]
            for index in df_proj['Closest Mesh Point Index']
        ])
        e1 = np.array([
            mesh.point_data['direction_1'][index]
            for index in df_proj['Closest Mesh Point Index']
        ])
        e2 = np.array([
            mesh.point_data['direction_2'][index]
            for index in df_proj['Closest Mesh Point Index']
        ])
        
        df_proj['normal vector Z'], df_proj['normal vector Y'], df_proj['normal vector X'] = normal_vectors[:,0],normal_vectors[:,1],normal_vectors[:,2]
        df_proj['e1 Z'], df_proj['e1 Y'], df_proj['e1 X'] = e1[:,0],e1[:,1],e1[:,2]
        df_proj['e2 Z'], df_proj['e2 Y'], df_proj['e2 X'] = e2[:,0],e2[:,1],e2[:,2]


        # Append the dataframe to the list
        all_dfs.append(df_proj)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

import pandas as pd
import numpy as np
import re
import statsmodels.formula.api as smf

def extract_day_from_folder(folder_name):
    match = re.match(r'^(\d{6,8})', str(folder_name))
    return int(match.group(1)) if match else None

def compute_hierarchical_icc(df, value_col):
    if df['day'].nunique() < 2 or df['folder_name'].nunique() < 2:
        return None, None, None, None
    try:
        model = smf.mixedlm(
            formula=f"{value_col} ~ 1",
            data=df,
            groups=df['day'],
            re_formula="1"
        )
        result = model.fit(reml=True)
        var_day = result.cov_re.iloc[0, 0]

        # Now folder-level as nested groups
        df['day_folder'] = df['day'].astype(str) + '_' + df['folder_name'].astype(str)
        model_nested = smf.mixedlm(
            formula=f"{value_col} ~ 1",
            data=df,
            groups=df['day_folder'],
            re_formula="1"
        )
        result_nested = model_nested.fit(reml=True)
        var_folder = result_nested.cov_re.iloc[0, 0]
        var_within = result_nested.scale

        total_var = var_day + var_folder + var_within
        icc_day = var_day / total_var
        icc_folder = var_folder / total_var
        total_icc = icc_day + icc_folder

        return total_icc, icc_day, icc_folder, var_within
    except:
        return None, None, None, None

def compute_design_effects(df, value_col, folder_col, group_cols):
    df = df.copy()
    df['day'] = df[folder_col].apply(extract_day_from_folder)

    results = []
    for keys, group_df in df.groupby(group_cols):
        total_icc, icc_day, icc_folder, var_w = compute_hierarchical_icc(group_df, value_col=value_col)
        N = len(group_df)
        n = group_df['day'].nunique()

        if total_icc is None or n <= 1:
            continue

        m_bar = N / n
        design_effect = 1 + (m_bar - 1) * total_icc
        N_eff = N / design_effect

        result_row = {col: val for col, val in zip(group_cols, keys)}
        result_row.update({
            'N': N,
            'n_days': n,
            'ICC Total': total_icc,
            'ICC Day': icc_day,
            'ICC Folder': icc_folder,
            'Design Effect': design_effect,
            'N_eff': N_eff,
            'Within Var': var_w
        })
        results.append(result_row)

    return pd.DataFrame(results)
import scipy.stats as stats
def plot_volume_histogram_by_condition(df, value_col='Volume', time_points=[96, 144], condition_col='condition', time_col='time in hpf', volume_threshold=600):
    fig, axes = plt.subplots(len(time_points), 1, figsize=(10, 6), sharex=True)
    colors = {'Development': 'blue', 'Regeneration': 'orange'}

    df = df[df[value_col] <= volume_threshold].copy()

    for i, t in enumerate(time_points):
        ax = axes[i] if len(time_points) > 1 else axes
        df_sub = df[df[time_col] == t].copy()
        df_sub['day'] = df_sub['folder_name'].apply(extract_day_from_folder)

        conds = df_sub[condition_col].unique()
        values = []
        means = {}
        neffs = {}
        stds = {}

        for cond in conds:
            vals = df_sub[df_sub[condition_col] == cond][value_col]
            n = len(vals)
            df_sub_group = df_sub[df_sub[condition_col] == cond].copy()
            total_icc, *_ = compute_hierarchical_icc(df_sub_group, value_col)
            m_bar = n / df_sub_group['day'].nunique() if df_sub_group['day'].nunique() > 1 else 1
            design_effect = 1 + (m_bar - 1) * total_icc if total_icc is not None else 1
            neff = n / design_effect if design_effect > 0 else n
            neffs[cond] = neff
            stds[cond] = vals.std()

            ax.hist(vals, bins=50, alpha=0.6, density=True, label=f'{cond} (N_eff={neff:.1f})', color=colors.get(cond, None))
            mu = vals.mean()
            ax.axvline(mu, linestyle='--', color=colors.get(cond, None))
            values.append(vals)
            means[cond] = mu

        if len(conds) == 2:
            try:
                mu_diff = means[conds[0]] - means[conds[1]]
                se_diff = np.sqrt(stds[conds[0]]**2 / neffs[conds[0]] + stds[conds[1]]**2 / neffs[conds[1]])
                t_stat = mu_diff / se_diff
                df_denom = min(neffs[conds[0]], neffs[conds[1]]) - 1
                p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df_denom))
                ax.text(0.95, 0.95, f'p = {p_val*100:.1f}%', transform=ax.transAxes,
                    ha='right', va='top', fontsize=12)
            except Exception as e:
                ax.text(0.95, 0.95, 't-test failed', transform=ax.transAxes,
                        ha='right', va='top', fontsize=12, color='red')

        ax.set_ylabel(f'{t} hpf')
        ax.legend()

    axes[-1].set_xlabel(value_col)
    fig.suptitle(f'{value_col} distribution at selected timepoints (<= {volume_threshold})')
    plt.tight_layout()
    plt.show()

def main():
    # merged_df=collect_dfs()
    # merged_df.to_csv('data.csv', index=False)
    prop_df = pd.read_csv('data.csv')
    # print(prop_df.columns)
    #proj_df=collect_dfs_proj()
    #proj_df.to_hdf('data_proj.h5', key='df', mode='w')
    proj_df = pd.read_hdf('data_proj.h5', key='df')
    # print(proj_df.columns)
    merged_df = pd.merge(prop_df, proj_df, on=['Label','time in hpf','condition'], how='inner')

    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
    merged_df = merged_df.dropna()


    merged_df['2I1/I2+I3']=2*merged_df['moments_eigvals 1']/(merged_df['moments_eigvals 2']+merged_df['moments_eigvals 3'])
    merged_df['elongation']=np.sqrt(merged_df['2I1/I2+I3'])
    merged_df['I1+I2/2I3']=(merged_df['moments_eigvals 1']+merged_df['moments_eigvals 2'])/(2*merged_df['moments_eigvals 3'])
    merged_df['I1/I2']=merged_df['moments_eigvals 1']/merged_df['moments_eigvals 2']
    merged_df['I2/I3']=merged_df['moments_eigvals 2']/merged_df['moments_eigvals 3']

    merged_df['orientation'] = (
    merged_df['eigvec1_X'] * merged_df['normal vector X'] +
    merged_df['eigvec1_Y'] * merged_df['normal vector Y'] +
    merged_df['eigvec1_Z'] * merged_df['normal vector Z']
    )*(
    merged_df['eigvec1_X'] * merged_df['normal vector X'] +
    merged_df['eigvec1_Y'] * merged_df['normal vector Y'] +
    merged_df['eigvec1_Z'] * merged_df['normal vector Z']
    )

    # merged_df['orientation'] = np.abs(
    # merged_df['eigvec1_X'] * merged_df['e2 X'] +
    # merged_df['eigvec1_Y'] * merged_df['e2 Y'] +
    # merged_df['eigvec1_Z'] * merged_df['e2 Z']
    # )

    print(merged_df.columns)
    print(merged_df.head())
    print(merged_df.shape)


    # df_summary = compute_design_effects(
    #     df=merged_df,
    #     value_col='Volume',
    #     folder_col='folder_name',
    #     group_cols=['time in hpf', 'condition']
    # )
    # print(df_summary)

    #plot_volume_histogram_by_condition(merged_df, value_col='Volume', time_points=[96, 144], condition_col='condition', time_col='time in hpf')

    # return
    plot_double_timeseries(merged_df, y_col='Volume', style='violin',y_scaling=1,y_name=r'Cell Volume $$\mu m^3$$',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='Surface Area', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='Solidity', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='Sphericity', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='elongation', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='I1+I2/2I3', style='violin',test_significance=True,y0=0,show_scatter=False)
    #plot_double_timeseries(merged_df, y_col='orientation', style='violin',test_significance=True,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='I1/I2', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='I2/I3', style='violin',test_significance=True,y0=0,show_scatter=False)
    return

    color_dict = {'Regeneration': 'orange',
                 'Development': 'blue', 
                 }
    marker_dict = {'Development': 'circle', 'Regeneration': 'triangle'}
    corner_plot = plot_scatter_corner(df=merged_df, parameters=['Volume','Surface Area','2I1/I2+I3', 'I1+I2/2I3','orientation'], color_col='time in hpf',color_dict=color_dict,marker_col='condition',marker_dict=marker_dict)
    show(corner_plot)
    corner_plot = plot_scatter_corner(df=merged_df, parameters=['Volume','Surface Area','2I1/I2+I3', 'I1+I2/2I3','orientation'], color_col='condition',color_dict=color_dict,marker_col='condition',marker_dict=marker_dict)
    show(corner_plot)

def draw_pictogram_solidity(solidity):
    """
    Draws a cross-shaped pictogram based on the given solidity.
    
    Parameters:
        solidity (float): The solidity value (between 0 and 1) to compute the extent 'h' of the cross arms.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # solidity = (1+4h)/(1+4h+2h²)
    a = 2 * solidity
    b = 4 * solidity - 4
    c = solidity - 1
    
    # Solve for h using the quadratic formula
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("Invalid solidity value. Discriminant is negative.")
    
    h = (-b + np.sqrt(discriminant)) / (2 * a)
    
    print((1+4*h)/(1+4*h+2*h*h))

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    

    # Define the sand color
    sand_color = '#C2B280'  # RGB Hex for a sand-like color
    
    # Draw the central square (1x1)
    central_square = patches.Rectangle((-0.5, -0.5), 1, 1, color=sand_color)
    ax.add_patch(central_square)
    
    # Draw the cross arms (4 rectangles extending outwards)
    arm_rectangles = [
        patches.Rectangle((-0.5, 0.5), 1, h, color=sand_color),  # Top rectangle
        patches.Rectangle((-0.5, -0.5 - h), 1, h, color=sand_color),  # Bottom rectangle
        patches.Rectangle((-0.5 - h, -0.5), h, 1, color=sand_color),  # Left rectangle
        patches.Rectangle((0.5, -0.5), h, 1, color=sand_color)  # Right rectangle
    ]
    
    # Add rectangles to the plot
    for rect in arm_rectangles:
        ax.add_patch(rect)
    ax.set_xlim(-1-h, 1+h)  # Set axis limits
    ax.set_ylim(-1-h, 1+h)
    ax.set_aspect('equal')  # Keep aspect ratio square
    # Remove axes for clean pictogram
    plt.axis('off')
    plt.title(f"Cross Pictogram for Solidity = {solidity:.2f}, h = {h:.2f}")
    plt.show()

def calculate_volume(a, b, c):
    """Calculate the volume of an ellipsoid."""
    return (4 / 3) * np.pi * a * b * c

from scipy.special import elliprd

def calculate_surface_area(a, b, c):
    """
    Calculate the surface area of a general ellipsoid using Carlson elliptic integrals.

    Parameters:
        a (float): Semi-major axis.
        b (float): Semi-intermediate axis.
        c (float): Semi-minor axis.

    Returns:
        float: Surface area of the ellipsoid.
    """
    if not (a >= b >= c):
        print(a,b,c)
        raise ValueError("Axes must satisfy a >= b >= c for this implementation.")


    # Compute Carlson's elliptic integral R_G
    R_G = elliprd(b*b/a/a, c*c/a/a, 1)  # Using scipy.special.elliprd for R_G
    # Surface area formula
    surface_area = 4 * np.pi * b*c * R_G
    return surface_area


def sphericity_function(c, a, b, sphericity):
    """
    Function to solve for the third axis (c) given sphericity.
    """
    print(a,b,c)
    volume = calculate_volume(a, b, c)
    r_equiv = (3 * volume / (4 * np.pi)) ** (1 / 3)  # Equivalent radius of a sphere
    surface_area = calculate_surface_area(c,b,a)
    print(surface_area)
    sphericity_value = (4 * np.pi * r_equiv**2) / surface_area
    return sphericity_value - sphericity

def draw_pictogram_sphericity(sphericity, a=1, b=1.01):
    """
    Draws a 3D ellipsoid given two smallest axes and a target sphericity.
    
    Parameters:
        sphericity (float): Target sphericity value.
        a (float): Smallest axis length.
        b (float): Second smallest axis length.
    """
    from scipy.optimize import least_squares
    import matplotlib.pyplot as plt
    # Solve for the largest axis c
    c_initial_guess = max(a, b) + 1  # Initial guess
    bounds = (max(a, b), np.inf)  # c must be larger than max(a, b)
    result = least_squares(sphericity_function, c_initial_guess, bounds=bounds, args=(a, b, sphericity))
    print(result)
    c_solution = result.x[0]
    # Validate result
    print(f"Calculated axis lengths: a = {a}, b = {b}, c = {c_solution:.4f}")
    
    # Generate ellipsoid surface
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c_solution * np.outer(np.ones_like(u), np.cos(v))
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(x, y, z, color='#C2B280', rstride=2, cstride=2, alpha=0.8, edgecolor='k')
    
    # Equal aspect ratio
    max_range = max(a, b, c_solution)
    ax.set_box_aspect([a / max_range, b / max_range, c_solution / max_range])  # Aspect ratio

    # Remove axes, grid, and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def inertia_residuals(axes, I_x, I_y, I_z, M):
    """
    Residual function to solve for ellipsoid axes given moments of inertia.
    
    Parameters:
        axes (list): Semi-axes [a, b, c].
        I_x, I_y, I_z (float): Moments of inertia.
        M (float): Mass of the ellipsoid.
    
    Returns:
        list: Residuals for the moments of inertia equations.
    """
    a, b, c = axes
    residual_x = I_x - (1/5) * M * (b**2 + c**2)
    residual_y = I_y - (1/5) * M * (a**2 + c**2)
    residual_z = I_z - (1/5) * M * (a**2 + b**2)
    return [residual_x, residual_y, residual_z]

def draw_ellipsoid_with_inertia(I_x, I_y, I_z, M=1.0):
    """
    Draws a 3D ellipsoid with given moments of inertia I_x, I_y, I_z.
    
    Parameters:
        I_x, I_y, I_z (float): Moments of inertia.
        M (float): Mass of the ellipsoid (default = 1.0).
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares
    # Initial guess for axes
    initial_guess = [1.0, 1.0, 1.0]
    
    # Solve for the axes using least_squares
    result = least_squares(inertia_residuals, initial_guess, args=(I_x, I_y, I_z, M), bounds=(0.1, np.inf))
    a, b, c = result.x
    print(f"Calculated semi-axes: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
    
    # Generate ellipsoid surface
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    
    # Plot the ellipsoid
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(x, y, z, color='#C2B280', rstride=2, cstride=2, alpha=0.8, edgecolor='k')
    
    # Equal aspect ratio
    max_range = max(a, b, c)
    ax.set_box_aspect([a / max_range, b / max_range, c / max_range])
    
    # Remove axes, grid, and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    
    # Title
    ax.set_title(f"Ellipsoid with Moments of Inertia: I_x={I_x}, I_y={I_y}, I_z={I_z}", fontsize=14)
    
    plt.tight_layout()
    plt.show()

def statistics():
    prop_df = pd.read_hdf('data.h5', key='df')
    proj_df = pd.read_hdf('data_proj.h5', key='df')

    merged_df = pd.merge(prop_df, proj_df, on=['Label','time in hpf','condition'], how='inner')

    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
    merged_df = merged_df.dropna()


    merged_df['2I1/I2+I3']=2*merged_df['moments_eigvals 1']/(merged_df['moments_eigvals 2']+merged_df['moments_eigvals 3'])
    merged_df['I1+I2/2I3']=(merged_df['moments_eigvals 1']+merged_df['moments_eigvals 2'])/(2*merged_df['moments_eigvals 3'])
    merged_df['I1/I2']=merged_df['moments_eigvals 1']/merged_df['moments_eigvals 2']
    merged_df['I2/I3']=merged_df['moments_eigvals 2']/merged_df['moments_eigvals 3']

    merged_df['orientation'] = (
    merged_df['eigvec1_X'] * merged_df['normal vector X'] +
    merged_df['eigvec1_Y'] * merged_df['normal vector Y'] +
    merged_df['eigvec1_Z'] * merged_df['normal vector Z']
    )*(
    merged_df['eigvec1_X'] * merged_df['normal vector X'] +
    merged_df['eigvec1_Y'] * merged_df['normal vector Y'] +
    merged_df['eigvec1_Z'] * merged_df['normal vector Z']
    )

    print(merged_df.columns)


    from sklearn.preprocessing import StandardScaler

    merged_df['condition'] = pd.factorize(merged_df['condition'])[0]

    # Select relevant columns
    selected_features = [
        'condition','time in hpf','Volume', 'Surface Area', 'Solidity', 'Sphericity',
        '2I1/I2+I3', 'I1+I2/2I3', 'orientation'
    ]
    data = merged_df[selected_features]

    # Scale numerical columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Convert back to DataFrame for easier handling
    scaled_df = pd.DataFrame(scaled_data, columns=selected_features)

    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.GraphUtils import GraphUtils
    #from causallearn.utils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    # Convert to numpy array for FCI
    data_array = scaled_df.to_numpy()

    # Create BackgroundKnowledge object
    bg_knowledge = BackgroundKnowledge()

    # Add background knowledge: 'condition' and 'time in hpf' have no parents
    condition_idx = scaled_df.columns.get_loc('condition')
    time_in_hpf_idx = scaled_df.columns.get_loc('time in hpf')

    cg, _ = fci(data_array, alpha=0.05, indep_test='fisherz')
    nodes=cg.get_nodes()
    print(nodes)

    for n in nodes:
        bg_knowledge.add_forbidden_by_node(n,nodes[condition_idx])
        bg_knowledge.add_forbidden_by_node(n,nodes[time_in_hpf_idx])

    # Function to apply the FCI algorithm with background knowledge
    def apply_fci(data, alpha=0.05, indep_test='fisherz', bg_knowledge=None):
        cg, _ = fci(data, alpha=alpha, indep_test=indep_test, background_knowledge=bg_knowledge)
        pyd_graph = GraphUtils.to_pydot(cg)
        return pyd_graph

    # Try different significance levels and independence tests
    significance_levels = [0.01, 0.05, 0.1]
    independence_tests = ['fisherz', 'kci']

    for test in independence_tests:
        for alpha in significance_levels:
            print(f"Running FCI with {test} test and alpha={alpha}")
            graph = apply_fci(data_array, alpha, test, bg_knowledge)
            graph.write_png(f"causal_graph_{test}_alpha{alpha}.png")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans,AffinityPropagation,MeanShift
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

def cluster():
    # Load data
    prop_df = pd.read_hdf('data.h5', key='df')
    
    #proj_df = pd.read_hdf('data_proj.h5', key='df')

    # Merge dataframes
    #merged_df = pd.merge(prop_df, proj_df, on=['Label', 'time in hpf', 'condition'], how='inner')
    prop_df = prop_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Calculate new features
    prop_df['2I1/I2+I3'] = 2 * prop_df['moments_eigvals 1'] / (prop_df['moments_eigvals 2'] + prop_df['moments_eigvals 3'])
    prop_df['I1+I2/2I3'] = (prop_df['moments_eigvals 1'] + prop_df['moments_eigvals 2']) / (2 * prop_df['moments_eigvals 3'])
    prop_df['I1/I2'] = prop_df['moments_eigvals 1'] / prop_df['moments_eigvals 2']
    prop_df['I2/I3'] = prop_df['moments_eigvals 2'] / prop_df['moments_eigvals 3']

    
    # Select features for scaling
    features_to_scale = [
        'Volume', 'Surface Area', 'Solidity', 'Sphericity',
        '2I1/I2+I3', 'I1+I2/2I3'
    ]

    # Scale only selected features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(prop_df[features_to_scale])

    # Create scaled DataFrame, keeping condition and time in hpf unchanged
    scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale)
    scaled_df['condition'] = prop_df['condition'].values
    scaled_df['time in hpf'] = prop_df['time in hpf'].values
    
    N1 = 2000  # Number of samples for times <= 72
    N2 = 2000  # Number of samples for times > 72
    # Split the DataFrame into two subsets based on 'time in hpf'
    subset_early = scaled_df[scaled_df['time in hpf'] <= 72].sample(n=N1, random_state=42)
    subset_late = scaled_df[scaled_df['time in hpf'] > 72].sample(n=N2, random_state=42)
    # Concatenate both subsets to form the final reduced DataFrame
    scaled_df = pd.concat([subset_early, subset_late])
    # Shuffle the combined DataFrame (optional, if order matters)
    scaled_df = scaled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # ---- Embedding Methods ----
    def apply_embedding(data, method='pca'):
        if method == 'pca':
            pca = PCA(n_components=2)
            embedding = pca.fit_transform(data)
        elif method == 'tsne':
            tsne = TSNE(n_components=2)
            embedding = tsne.fit_transform(data)
        elif method == 'umap':
            umap_reducer = umap.UMAP(n_components=2)
            embedding = umap_reducer.fit_transform(data)
        elif method == 'isomap':
            isomap = Isomap(n_components=2)
            embedding = isomap.fit_transform(data)
        elif method == 'lle':
            lle = LocallyLinearEmbedding(n_components=2)
            embedding = lle.fit_transform(data)
        elif method == 'specemb':
            specemb = SpectralEmbedding(n_components=2)
            embedding = specemb.fit_transform(data)
        elif method == 'mds':
            mds = MDS(n_components=2)
            embedding = mds.fit_transform(data)
        else:
            raise ValueError("Unknown embedding method")
        return embedding

    # Apply embeddings
    features = scaled_df.drop(['condition', 'time in hpf'], axis=1)
    embeddings = {
        'PCA': apply_embedding(features, method='pca'),
        #'t-SNE': apply_embedding(features, method='tsne'),
        'UMAP': apply_embedding(features, method='umap'),
        #'ISOMAP': apply_embedding(features, method='isomap'),
        #'LLE': apply_embedding(features, method='lle'),
        'SPECEMB': apply_embedding(features, method='specemb'),
        #'MDS' : apply_embedding(features, method='mds')
    }

    # ---- Plot Embeddings with color-coded 'time in hpf' and marker style by 'condition' ----
    # Get unique values in 'time in hpf'
    unique_times = sorted(scaled_df['time in hpf'].unique())

    # Create a discrete color palette with one color for each unique time in hpf
    color_palette = sns.color_palette("flare", len(unique_times))

    # Create a mapping from 'time in hpf' to colors
    time_color_map = {time: color for time, color in zip(unique_times, color_palette)}

    for embed_name, embed_data in embeddings.items():
        plt.figure(figsize=(10, 6))

        # Create scatter plot with color based on 'time in hpf' and marker style by 'condition'
        sns.scatterplot(
            x=embed_data[:, 0], y=embed_data[:, 1],
            hue=scaled_df['time in hpf'],  # Color-coded by 'time in hpf'
            style=scaled_df['condition'],  # Marker style by 'condition'
            palette=time_color_map,  # Custom color palette
            markers=True,  # Enable different markers
            edgecolor='w',  # Add white edge to markers for better visibility
            s=50  # Marker size
        )

        # Set plot title and labels
        plt.title(f'{embed_name} Embedding')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        # Customize legend
        plt.legend(title='Time in hpf & Condition', loc='upper left')

        # Show plot
        plt.tight_layout()
        plt.show()

    
    # ---- Clustering Methods ----
    def apply_clustering(data, method, n_clusters=None):
        if method == 'kmeans':
            return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data)
        elif method == 'agglomerative':
            return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(data)
        elif method == 'gmm':
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            return gmm.fit_predict(data)
        elif method == 'ap':
            ap = AffinityPropagation(damping=0.75,random_state=42)
            return ap.fit_predict(data)
        elif method == 'ms':
            sm = MeanShift()
            return sm.fit_predict(data)
        else:
            raise ValueError("Unknown clustering method")

    # Define clustering methods and cluster numbers
    clustering_methods = {
        'KMeans': [2, 3, 4],
        'Agglomerative': [2, 3, 4],
        'GMM': [2, 3, 4],
        'AP': [None],
        'MS': [None]
    }

    # Apply clustering on original feature data
    clustering_results = {}
    for method, clusters in clustering_methods.items():

        for n_cluster in clusters:
            labels = apply_clustering(features, method.lower(), n_cluster)
            clustering_results[f'{method} {n_cluster} clusters'] = labels
   

    # ---- Plot Clustering Results on Embeddings ----
    for embed_name, embed_data in embeddings.items():
        for method, labels in clustering_results.items():
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=embed_data[:, 0], y=embed_data[:, 1], hue=labels, palette='viridis', legend='full')
            plt.title(f'{method} Clustering ({embed_name} Embedding)')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend(title='Cluster')
            plt.show()



if __name__ == "__main__":
    main()

    # draw_pictogram_solidity(0.9)
    # draw_pictogram_solidity(0.8)
    # draw_pictogram_solidity(0.7)
    #draw_pictogram_sphericity(0.6)
    #draw_ellipsoid_with_inertia(I_x=1.0, I_y=1.0, I_z=0.1)

    #statistics()
    #cluster()