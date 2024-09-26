import os
import pandas as pd
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import spatial_efd
import gmsh
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from boundary import getBoundary
from config import FlatFin_path
from IO import get_JSON

def getPolygon(mesh_3d: pv.PolyData):
    boundary_indices = getBoundary(mesh_3d)
    
    bc1 = mesh_3d.point_data['coord_1'][boundary_indices]
    bc2 = mesh_3d.point_data['coord_2'][boundary_indices]
    
    centroid = centroid_Polygon(bc1, bc2)
    b = np.stack((bc1, bc2))
    #print(b.shape)
    cross_prod = (bc1[0] - centroid[0]) * (bc2[1] - centroid[1]) - (bc2[0] - centroid[1]) * (bc1[1] - centroid[0])
    if cross_prod <0:
        b=b[:,::-1]
        #print('invert')
    return b

def centroid_Polygon(x, y):
    # Ensure that the polygon is closed by appending the first point to the end
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # Calculate the signed area (A) of the polygon
    A = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    # Calculate the centroid coordinates
    C_x = (1 / (6 * A)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    C_y = (1 / (6 * A)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))

    return np.array([C_x, C_y])

def plot_grouped_polygons_and_average(df, all_poly, all_coeff, harmonics=2):
    grouped = df.groupby(['time in hpf', 'condition'])

    for group, indices in grouped.groups.items():
        group_polygons = [all_poly[df['folder_name'].iloc[i]] for i in indices]
        group_coeffs = [all_coeff[df['folder_name'].iloc[i]] for i in indices]

        # Compute the average Fourier coefficients for the group
        avg_coeff = np.mean(group_coeffs, axis=0)
        avg_coeff = spatial_efd.AverageCoefficients(group_coeffs)
        print(group_coeffs)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot individual polygons (centered by subtracting the centroid)
        for coeff in group_coeffs:
            x, y = spatial_efd.inverse_transform(coeff, harmonic=harmonics,n_coords=20)
            ax.plot(x, y, 'r', alpha=0.3, label='Individual Polygons smooth ' if not ax.lines else "")
        
        for polygon in group_polygons:
            x, y = polygon[0, :], polygon[1, :]
            centroid = centroid_Polygon(x, y)
            x_centered = x - centroid[0]
            y_centered = y - centroid[1]
            ax.plot(x_centered, y_centered, 'g', alpha=0.3, label='Individual Polygons' if not ax.lines else "")
        
        # Inverse transform the average coefficients to reconstruct the averaged polygon
        xt, yt = spatial_efd.inverse_transform(avg_coeff, harmonic=harmonics,n_coords=20)
        
        # Plot the averaged polygon (centered at the origin)
        ax.plot(xt, yt, 'b', label='Averaged Polygon')

        ax.legend()
        ax.set_title(f"Group: {group}")
        plt.show()

def shift_coeff(coeff, theta):
    coeff_shifted = coeff.copy()  # Create a copy to avoid modifying the original coefficients
    num_orders = coeff.shape[0]   # Assuming coeff has shape (n_harmonics, 4) for 2x2 matrices
    
    # Loop through each order n
    for n in range(0, num_orders):  
        # Get the phase shift matrix for order n
        cos_ntheta = np.cos((n+1)* theta)
        sin_ntheta = np.sin((n+1)* theta)
        phase_shift_matrix = np.array([[cos_ntheta, sin_ntheta],
                                       [-sin_ntheta, cos_ntheta]])
        
        # Reshape the coefficients to a 2x2 matrix
        coeff_matrix = coeff[n].reshape(2, 2).T
        
        # Apply the phase shift matrix
        coeff_shifted[n] = (phase_shift_matrix @ coeff_matrix).T.flatten()  # Flatten back to a 1D array
    
    return coeff_shifted

def getCoeff(polygon, harmonics):
    # Extract x and y coordinates from the polygon (2xN)
    x, y = polygon[0, :], polygon[1, :]

    # Calculate Fourier descriptors
    X,Y=spatial_efd.CloseContour(x,y)
    coeffs = spatial_efd.CalculateEFD(X, Y, harmonics=harmonics)
    
    return coeffs

def get_avg_Shapes(df, all_coeff, harmonics=2):
    grouped = df.groupby(['time in hpf', 'condition'])

    avgCoeffs={}
    avgPolygons={}

    for group, indices in grouped.groups.items():
        group_coeffs = [all_coeff[df['folder_name'].iloc[i]] for i in indices]

        # Compute the average Fourier coefficients for the group
        avg_coeff = spatial_efd.AverageCoefficients(group_coeffs)
        avgCoeffs[group] = avg_coeff
       
        # Inverse transform the average coefficients to reconstruct the averaged polygon
        xt, yt = spatial_efd.inverse_transform(avg_coeff, harmonic=harmonics,n_coords=20)
        avgPolygons[group]=np.stack((xt,yt))

    return avgCoeffs,avgPolygons   

import matplotlib.cm as cm
def plot_time_evolution(avgPolygons):
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Normalize the time values for color mapping
    times = np.array([key[0] for key in avgPolygons.keys()])
    min_time, max_time = times.min(), times.max()
    norm = plt.Normalize(min_time, max_time)
    cmap = cm.viridis  # Colormap for time-based coloring

    for (time, condition), polygon in avgPolygons.items():
        # Extract x and y coordinates
        x, y = polygon[0, :], polygon[1, :]
        
        # Choose color based on time and line style based on condition
        color = cmap(norm(time))
        if condition == 'Development':
            linestyle = '-'  # Solid line for Development
        elif condition == 'Regeneration':
            linestyle = '--'  # Dashed line for Regeneration
        
        # Plot the polygon
        ax.plot(x, y, linestyle=linestyle, color=color, label=f'Time: {time}, {condition}')

    # Create a colorbar to represent the time evolution
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(times)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time in hpf')
    
    # Set plot labels and title
    ax.set_title("Time Evolution of Polygons")
    ax.set_xlabel("X Coordinates")
    ax.set_ylabel("Y Coordinates")
    
    # Show plot
    plt.show()

def interpolatePolygons(avgPolygons):
    def linear_interpolate(polygon1, polygon2, t1, t2, t):
        """Interpolate linearly between two polygons for a given time t."""
        factor = (t - t1) / (t2 - t1)
        interpolated_polygon = polygon1 + factor * (polygon2 - polygon1)
        return interpolated_polygon

    # Separate polygons by condition
    conditions = {'Development': {}, 'Regeneration': {}}
    
    for (time, condition), polygon in avgPolygons.items():
        conditions[condition][time] = polygon

    # Interpolate polygons for both conditions
    for condition, poly_dict in conditions.items():
        times = sorted(poly_dict.keys())  # Sort time steps
        min_time, max_time = times[0], times[-1]
        
        # Create new polygons for missing times
        for t in range(min_time, max_time):
            if t not in poly_dict:
                # Find the closest surrounding polygons for interpolation
                t1 = max([ti for ti in times if ti <= t])
                t2 = min([ti for ti in times if ti > t])
                
                polygon1 = poly_dict[t1]
                polygon2 = poly_dict[t2]
                
                # Interpolate between polygon1 and polygon2 for time t
                interpolated_polygon = linear_interpolate(polygon1, polygon2, t1, t2, t)
                
                # Add the interpolated polygon to the dictionary
                avgPolygons[(t, condition)] = interpolated_polygon

    # Ensure the avgPolygons dictionary now has entries for all time steps
    return avgPolygons
from matplotlib.animation import FuncAnimation
def get_xy_lims(avgPolygons):
    min_val, max_val = float('inf'), float('-inf')  # Initialize with extreme values
    
    # Iterate through all polygons in avgPolygons
    for _, polygon in avgPolygons.items():
        x_values = polygon[0, :]  # Extract x-coordinates
        y_values = polygon[1, :]  # Extract y-coordinates
        
        # Find the min and max of both x and y
        min_val = min(min_val, np.min(x_values), np.min(y_values))
        max_val = max(max_val, np.max(x_values), np.max(y_values))
    
    return min_val, max_val

def movie_time_evolution(avgPolygons, save_path=None):
    # Get unique times and sort them
    times = sorted(list(set([key[0] for key in avgPolygons.keys()])))
    
    # Normalize the time values for color mapping
    min_time, max_time = min(times), max(times)
    norm = plt.Normalize(min_time, max_time)
    cmap = cm.viridis  # Color map based on time
    
    fig, ax = plt.subplots(figsize=(10, 10))

    # Initialize plot elements
    line_dev, = ax.plot([], [], 'g-', lw=2, label="Development")
    line_reg, = ax.plot([], [], 'b--', lw=2, label="Regeneration")
    
    # Set axis labels
    ax.set_xlabel("X Coordinates")
    ax.set_ylabel("Y Coordinates")
    ax.set_title("Time Evolution of Polygons")
    
    # Set fixed axis limits (adjust based on the polygon sizes)
    min_val, max_val=get_xy_lims(avgPolygons)
    ax.set_xlim(min_val-10, max_val+10)
    ax.set_ylim(min_val-10, max_val+10)
    
    # Create a color bar for the time evolution
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(times)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time in hpf')

    def init():
        """Initial setup of the plot."""
        return line_dev, line_reg

    def update(frame):
        """Update the plot for each frame of the animation."""
        time = times[frame]
        
        # Get the polygons corresponding to the current time step
        polygons_to_plot = [(condition, polygon) for (t, condition), polygon in avgPolygons.items() if t == time]

        print(f"Time: {time}, Polygons to Plot: {len(polygons_to_plot)}")  # Debugging line

        # Loop over the polygons for this time step and update the plot
        for condition, polygon in polygons_to_plot:
            x, y = polygon[0, :], polygon[1, :]  # Extract x and y coordinates
            color = cmap(norm(time))

            print(f"Condition: {condition}, X: {x[:5]}, Y: {y[:5]}")  # Debugging line to print first few points
            
            if condition == 'Development':
                line_dev.set_data(x, y)  # Update development polygon data
                line_dev.set_color(color)  # Color it based on the current time
                
            elif condition == 'Regeneration':
                line_reg.set_data(x, y)  # Update regeneration polygon data
                line_reg.set_color(color)  # Color it based on the current time

        # Update the title to reflect the current time
        ax.set_title(f"Time: {time} hpf")

        return line_dev, line_reg

    # Create the animation without blit for proper frame updating
    anim = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False, interval=100)

    if save_path:
        # Save the animation as a video file
        anim.save(save_path, writer='ffmpeg', fps=2)
    else:
        # Display the animation
        plt.show()

def generate_2d_mesh(boundary_points, mesh_size=0.1):
    # Initialize Gmsh
    gmsh.initialize()

    # Create a new Gmsh model
    gmsh.model.add("2D Mesh")

    # Step 1: Define boundary points as Gmsh points
    point_tags = []
    for i, (x, y) in enumerate(boundary_points):
        point_tag = gmsh.model.geo.addPoint(x, y, 0, mesh_size)
        point_tags.append(point_tag)
    
    # Step 2: Create boundary lines from points and close the loop
    line_tags = []
    num_points = len(point_tags)
    for i in range(num_points):
        # Create a line between consecutive points
        line_tag = gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % num_points])
        line_tags.append(line_tag)

    # Step 3: Create a curve loop from the boundary lines
    curve_loop = gmsh.model.geo.addCurveLoop(line_tags)

    # Step 4: Create a plane surface bounded by the curve loop
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # Step 5: Synchronize the CAD representation with the Gmsh model
    gmsh.model.geo.synchronize()

    # Step 6: Generate the 2D mesh
    gmsh.model.mesh.generate(2)

    # Step 7: Retrieve mesh data (vertices and triangles)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)[:, :2]  # Extract x, y coordinates only

    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(2)
    triangles = np.array(element_node_tags[0]).reshape(-1, 3) - 1  # Convert to 0-based indexing

    # Finalize the gmsh session
    gmsh.finalize()

    return nodes, triangles


def get_boundary_indices(nodes, boundary_points):
    """
    Find the indices in the `nodes` array that correspond to the `boundary_points`.
    This uses a proximity check to account for floating-point differences.
    """
    boundary_indices = []
    for boundary_point in boundary_points:
        # Find the index of the node that is closest to the boundary point
        distances = np.linalg.norm(nodes - boundary_point, axis=1)
        closest_index = np.argmin(distances)
        boundary_indices.append(closest_index)
    
    return np.array(boundary_indices)

def get_last_time_meshes(avgPolygons, mesh_size=0.1):
    # Find the maximum time for each condition
    last_dev_time = max([time for (time, condition) in avgPolygons if condition == 'Development'])
    last_reg_time = max([time for (time, condition) in avgPolygons if condition == 'Regeneration'])
    
    # Extract the corresponding polygons
    dev_polygon = avgPolygons[(last_dev_time, 'Development')]
    reg_polygon = avgPolygons[(last_reg_time, 'Regeneration')]
    
    # Generate meshes for both polygons
    dev_nodes, dev_triangles = generate_2d_mesh(dev_polygon.T, mesh_size)
    reg_nodes, reg_triangles = generate_2d_mesh(reg_polygon.T, mesh_size)
    
    # Find the indices of the original boundary points in the new nodes array
    dev_boundary_indices = get_boundary_indices(dev_nodes, dev_polygon.T)
    reg_boundary_indices = get_boundary_indices(reg_nodes, reg_polygon.T)
    
    # Return meshes and boundary indices
    return (dev_nodes, dev_triangles, dev_boundary_indices), (reg_nodes, reg_triangles, reg_boundary_indices)

# Plotting function to visualize the mesh
def plot_mesh(nodes, triangles, ax, title):
    ax.triplot(nodes[:, 0], nodes[:, 1], triangles, color='blue')
    ax.scatter(nodes[:, 0], nodes[:, 1], color='red', s=10)
    ax.set_title(title)
    ax.set_aspect('equal')

def compute_element_stiffness_matrix(vertices):
    # Compute the area of the triangle (vertices are in counterclockwise order)
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    
    area = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    # Compute the gradients of the linear basis functions
    b1, c1 = y2 - y3, x3 - x2
    b2, c2 = y3 - y1, x1 - x3
    b3, c3 = y1 - y2, x2 - x1

    B = np.array([[b1, b2, b3], [c1, c2, c3]])

    # Local stiffness matrix
    local_K = (B.T @ B) / (4 * area)

    return local_K

def assemble_global_stiffness_matrix(nodes, triangles):
    num_nodes = len(nodes)
    K = sp.lil_matrix((num_nodes, num_nodes))  # Initialize global stiffness matrix

    for triangle in triangles:
        # Extract the vertices of the triangle
        vertices = nodes[triangle]

        # Compute the local stiffness matrix for this element
        local_K = compute_element_stiffness_matrix(vertices)

        # Assemble the local stiffness matrix into the global matrix
        for i, vi in enumerate(triangle):
            for j, vj in enumerate(triangle):
                K[vi, vj] += local_K[i, j]

    return K.tocsr()

def solve_laplace(nodes, triangles, boundary_indices, boundary_displacement):
    """
    Solves the Laplace equation using the finite element method to propagate
    boundary displacements to interior nodes.
    """
    num_nodes = len(nodes)

    # Assemble the global stiffness matrix
    K = assemble_global_stiffness_matrix(nodes, triangles)

    # Initialize the displacement vector for all nodes
    displacement = np.zeros((num_nodes, 2))

    # Apply boundary displacements
    displacement[boundary_indices] = boundary_displacement

    # Separate free nodes (interior) from boundary nodes
    free_indices = np.setdiff1d(np.arange(num_nodes), boundary_indices)

    # Solve for displacement at interior nodes using Laplace equation
    for dim in range(2):  # x and y dimensions
        # Construct the right-hand side (forcing) vector
        rhs = -K[free_indices, :][:, boundary_indices] @ displacement[boundary_indices, dim]
        
        # Solve for interior displacements
        K_free = K[free_indices, :][:, free_indices]  # Submatrix for free nodes
        displacement[free_indices, dim] = spsolve(K_free, rhs)

    return displacement

def plot_mesh_movement(nodes, triangles, boundary_indices, title):
    """
    Helper function to plot mesh movement for debugging purposes.
    """
    plt.figure(figsize=(6, 6))
    plt.triplot(nodes[:, 0], nodes[:, 1], triangles, color='blue')
    plt.scatter(nodes[boundary_indices, 0], nodes[boundary_indices, 1], color='red', label='Boundary Points')
    plt.title(title)
    plt.legend()
    plt.show()

def interpolate_mesh_over_time(avgPolygons, last_time_meshes):
    # Initialize the interpolated nodes dictionary for both conditions
    interpolated_nodes = {}

    # Define the conditions
    conditions = ['Development', 'Regeneration']

    # Loop over each condition (Development and Regeneration)
    for condition in conditions:
        # Extract the corresponding nodes, triangles, and boundary indices
        nodes, triangles, boundary_indices = last_time_meshes[condition]

        # Get the time steps for the current condition
        times = sorted([time for (time, cond) in avgPolygons if cond == condition])

        # Initialize the interpolation with the last time step
        interpolated_nodes[(times[-1], condition)] = nodes.copy()

        # Step backward from the last time to the first, computing displacement at each step
        for t_idx in range(len(times) - 2, -1, -1):
            current_time = times[t_idx]

            # Get the corresponding polygon for the current time step
            polygon = avgPolygons[(current_time, condition)].T

            # Compute the displacement for the boundary nodes
            boundary_displacement = polygon - nodes[boundary_indices]

            # Solve Laplace equation to propagate the displacement to the interior nodes
            displacement = solve_laplace(nodes, triangles, boundary_indices, boundary_displacement)

            # Apply the displacement to the nodes
            nodes += displacement

            # Store the interpolated node positions
            interpolated_nodes[(current_time, condition)] = nodes.copy()

    return interpolated_nodes


def get_mesh_lims(interpolated_dev_nodes, interpolated_reg_nodes):
    min_val, max_val = float('inf'), float('-inf')  # Initialize with extreme values
    
    # Iterate through all time steps for development nodes
    for nodes in interpolated_dev_nodes.values():
        x_values = nodes[:, 0]  # Extract x-coordinates
        y_values = nodes[:, 1]  # Extract y-coordinates
        
        # Find the min and max of both x and y
        min_val = min(min_val, np.min(x_values), np.min(y_values))
        max_val = max(max_val, np.max(x_values), np.max(y_values))
    
    # Iterate through all time steps for regeneration nodes
    for nodes in interpolated_reg_nodes.values():
        x_values = nodes[:, 0]  # Extract x-coordinates
        y_values = nodes[:, 1]  # Extract y-coordinates
        
        # Find the min and max of both x and y
        min_val = min(min_val, np.min(x_values), np.min(y_values))
        max_val = max(max_val, np.max(x_values), np.max(y_values))
    
    return min_val, max_val

def movie_grid_time_evolution(interpolated_nodes, triangles, save_path=None):
    # Get unique times sorted
    times = sorted(list(set([key[0] for key in interpolated_nodes.keys()])))
   
    # Create the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Set axis labels and titles
    ax1.set_title("Development Mesh")
    ax1.set_xlabel("X Coordinates")
    ax1.set_ylabel("Y Coordinates")
    ax2.set_title("Regeneration Mesh")
    ax2.set_xlabel("X Coordinates")
    ax2.set_ylabel("Y Coordinates")

    # Get the mesh limits for both conditions
    min_val, max_val = get_mesh_lims(
        {key: nodes for key, nodes in interpolated_nodes.items() if key[1] == 'Development'},
        {key: nodes for key, nodes in interpolated_nodes.items() if key[1] == 'Regeneration'}
    )
    min_val -= 10
    max_val += 10
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)

    def update(frame):
        time = times[frame]
        """Update the plots for each frame of the animation"""
        # Clear the axes before each update
        ax1.clear()
        ax2.clear()

        # Set titles again after clearing
        ax1.set_title(f"Development Mesh at Time {time}")
        ax2.set_title(f"Regeneration Mesh at Time {time}")
        
        ax1.set_xlabel("X Coordinates")
        ax1.set_ylabel("Y Coordinates")
        ax2.set_xlabel("X Coordinates")
        ax2.set_ylabel("Y Coordinates")

        # Set axis limits
        ax1.set_xlim(min_val, max_val)
        ax1.set_ylim(min_val, max_val)
        ax2.set_xlim(min_val, max_val)
        ax2.set_ylim(min_val, max_val)

        # Get the current node positions for development and regeneration
        dev_nodes = interpolated_nodes[(time, 'Development')]
        reg_nodes = interpolated_nodes[(time, 'Regeneration')]

        # Plot the development mesh
        ax1.triplot(dev_nodes[:, 0], dev_nodes[:, 1], triangles['Development'], color='blue')

        # Plot the regeneration mesh
        ax2.triplot(reg_nodes[:, 0], reg_nodes[:, 1], triangles['Regeneration'], color='blue')

        return ax1, ax2

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(times), blit=False, interval=200)

    if save_path:
        # Save the animation as a video file
        anim.save(save_path, writer='ffmpeg', fps=2)
    else:
        # Display the animation
        plt.show()

def main():
    n_harmonics=10

    FlatFin_folder_list = os.listdir(FlatFin_path)
    FlatFin_folder_list = [item for item in FlatFin_folder_list if os.path.isdir(os.path.join(FlatFin_path, item))]
    
    data_list = []
    all_coeff = {}
    all_poly = {}
    
    for FlatFin_folder in FlatFin_folder_list:
        FlatFin_dir_path = os.path.join(FlatFin_path, FlatFin_folder)
        MetaData = get_JSON(FlatFin_dir_path)
        
        if 'Thickness_MetaData' not in MetaData:
            continue
        
        MetaData = MetaData['Thickness_MetaData']
        
        data_list.append({
            'folder_name': FlatFin_folder,
            'file_name': MetaData['Surface file'],
            'condition': MetaData['condition'],
            'time in hpf': MetaData['time in hpf'],
            'genotype': MetaData['genotype'],
            'experimentalist': MetaData['experimentalist']
        })
        
        surface_file_name = MetaData['Surface file']
        mesh_3d = pv.read(os.path.join(FlatFin_dir_path, surface_file_name))
        polygon = getPolygon(mesh_3d)
        coeff = getCoeff(polygon, n_harmonics)


        # x, y = polygon[0, :], polygon[1, :]
        # centroid = centroid_Polygon(x, y)
        # x_centered = x - centroid[0]
        # y_centered = y - centroid[1]
        #plt.plot(x_centered, y_centered, 'g', alpha=0.3, label='Individual Polygons')
        xt, yt = spatial_efd.inverse_transform(coeff, harmonic=n_harmonics)
        #plt.plot(xt, yt, 'b', label='smooth Polygon')
        #print(coeff)
        x0=np.sum(coeff[:,2])
        y0=np.sum(coeff[:,0])
        #print(np.sum(coeff,axis=0))
        #plt.scatter((x0,x0), (y0,y0), label='start')
        theta=np.arctan2(y0, x0)
        coeff_shift=shift_coeff(coeff,-theta)
        #xt, yt = spatial_efd.inverse_transform(coeff_shift, harmonic=n_harmonics)
        #plt.plot(xt, yt, 'g', label='shift Polygon')
        # x0=np.sum(coeff_shift[:,2])
        # y0=np.sum(coeff_shift[:,0])
        #print(np.sum(coeff_shift,axis=0))
        #plt.scatter((x0,x0), (y0,y0), label='start')
       
        #plt.show()
        

        all_coeff[FlatFin_folder] = coeff_shift
        all_poly[FlatFin_folder] = polygon

    # Store the metadata in a pandas DataFrame
    df = pd.DataFrame(data_list)
    
    # Plot grouped polygons and the averaged Fourier descriptors
    #plot_grouped_polygons_and_average(df, all_poly, all_coeff,harmonics=n_harmonics)
    avgCoeffs,avgPolygons=get_avg_Shapes(df,all_coeff,n_harmonics)
    #plot_time_evolution(avgPolygons)
    interpolatePolygons(avgPolygons)
    #plot_time_evolution(avgPolygons)
    #movie_time_evolution(avgPolygons)

    (dev_nodes, dev_triangles, dev_boundary_indices), (reg_nodes, reg_triangles, reg_boundary_indices) = get_last_time_meshes(avgPolygons, mesh_size=1000.0)
    last_time_meshes = {
        'Development': (dev_nodes, dev_triangles, dev_boundary_indices),
        'Regeneration': (reg_nodes, reg_triangles, reg_boundary_indices)
    }
    triangles = {
        'Development': dev_triangles,
        'Regeneration': reg_triangles
    }
    interpolated_nodes=interpolate_mesh_over_time(avgPolygons, last_time_meshes)
    movie_grid_time_evolution(interpolated_nodes, triangles)





if __name__ == "__main__":
    main()