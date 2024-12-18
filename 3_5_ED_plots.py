import os
import pandas as pd
from plotBokehHelper import plot_double_timeseries
from plotHelper import plot_scatter_corner
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



def main():
    #merged_df=collect_dfs()
    #merged_df.to_hdf('data.h5', key='df', mode='w')
    prop_df = pd.read_hdf('data.h5', key='df')
   
    #proj_df=collect_dfs_proj()
    #proj_df.to_hdf('data_proj.h5', key='df', mode='w')
    proj_df = pd.read_hdf('data_proj.h5', key='df')

    merged_df = pd.merge(prop_df, proj_df, on=['Label','time in hpf','condition'], how='inner')

    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
    merged_df = merged_df.dropna()


    merged_df['2I1/I2+I3']=2*merged_df['moments_eigvals 1']/(merged_df['moments_eigvals 2']+merged_df['moments_eigvals 3'])
    merged_df['I1+I2/2I3']=(merged_df['moments_eigvals 1']+merged_df['moments_eigvals 2'])/(2*merged_df['moments_eigvals 3'])
    merged_df['I1/I2']=merged_df['moments_eigvals 1']/merged_df['moments_eigvals 2']
    merged_df['I2/I3']=merged_df['moments_eigvals 2']/merged_df['moments_eigvals 3']

    merged_df['orientation'] = np.abs(
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
    # print(merged_df)
    # return
    # plot_double_timeseries(merged_df, y_col='Volume', style='violin',y_scaling=1,y_name=r'Cell Volume $$\mu m^3$$',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='Surface Area', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='Solidity', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='Sphericity', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='2I1/I2+I3', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='I1+I2/2I3', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='orientation', style='violin',test_significance=True,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='I1/I2', style='violin',test_significance=True,y0=0,show_scatter=False)
    # plot_double_timeseries(merged_df, y_col='I2/I3', style='violin',test_significance=True,y0=0,show_scatter=False)
    

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


if __name__ == "__main__":
    main()

    # draw_pictogram_solidity(0.9)
    # draw_pictogram_solidity(0.8)
    # draw_pictogram_solidity(0.7)
    #draw_pictogram_sphericity(0.6)
    #draw_ellipsoid_with_inertia(I_x=1.0, I_y=1.0, I_z=0.1)