
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import re

# Function to filter outliers
def filter_outliers(data, n_std=3):
    mean = np.mean(data)
    std = np.std(data)
    return np.where(np.abs(data - mean) > n_std * std, np.nan, data)

# Function to plot 2D data using Matplotlib
def plot2d(ax, mesh, key, nstd=3):
    coord_1 = mesh.point_data['coord_1']
    coord_2 = mesh.point_data['coord_2']
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]
    triangulation = tri.Triangulation(coord_1, coord_2, faces)
    
    data = mesh.point_data[key]
    filtered_data = filter_outliers(data, nstd)
    
    # Plot the triangulation with the filtered data as color
    coloring = ax.tripcolor(triangulation, filtered_data, shading='flat', cmap='viridis')
    ax.set_xlabel('Coord 1')
    ax.set_ylabel('Coord 2')
    ax.set_aspect('equal')
    ax.grid(True)
    
    return coloring

# Function to extract category and time from folder names using regex
def extract_category_and_time(folder_name):
    pattern = r'([a-zA-Z]+)_(\d+)'
    match = re.match(pattern, folder_name)
    if match:
        category = match.group(1)
        time = int(match.group(2))
        return category, time
    return None, None

# Function to plot histograms sequentially for each category/time pair
def plotHistogramsSequentially(times, categories, meshes, keys, nstds):
    for i, mesh in enumerate(meshes):
        category, time = categories[i], times[i]
        fig, axs = plt.subplots(1, len(keys), figsize=(15, 5))
        fig.suptitle(f'{category} - Time {time} hpf')

        for ax, key, nstd in zip(axs, keys, nstds):
            print(f"Plotting {key} for {category} at time {time}")
            coloring = plot2d(ax, mesh, key, nstd)
            fig.colorbar(coloring, ax=ax, orientation='vertical', label=key)

        plt.show()

# Function to compare histograms of all meshes for the same key at once
def plotHistogramsComparison(times, categories, meshes, keys, nstds):
    for key, nstd in zip(keys, nstds):
        print(f"Comparing {key} across all meshes")
        fig, axs = plt.subplots(2, len(meshes) // 2, figsize=(20, 10))
        axs = axs.flatten()

        for i, mesh in enumerate(meshes):
            category, time = categories[i], times[i]
            ax = axs[i]
            coloring = plot2d(ax, mesh, key, nstd)
            ax.set_title(f'{category} - {time} hpf')

        fig.colorbar(coloring, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
        fig.suptitle(f'Comparison of {key}')
        plt.show()

from numpy.linalg import eig
# Function to calculate eigenvalues and eigenvectors from a 2x2 tensor
def calculate_eigenvectors(tensor_data):
    n_points = len(tensor_data) // 4
    eigenvalues = np.zeros((n_points, 2))
    eigenvectors = np.zeros((n_points, 2, 2))
    
    for i in range(n_points):
        tensor = np.array([[tensor_data[i * 4], tensor_data[i * 4 + 1]],
                           [tensor_data[i * 4 + 2], tensor_data[i * 4 + 3]]])
        # Eigenvalue and eigenvector calculation
        w, v = eig(tensor)
        eigenvalues[i] = w
        eigenvectors[i] = v

    return eigenvalues, eigenvectors

# Function to plot eigenvectors
def plot_eigenvectors(ax, mesh, eigenvalues, eigenvectors, scale=0.1):
    coord_1 = mesh.point_data['coord_1']
    coord_2 = mesh.point_data['coord_2']
    
    for i in range(len(coord_1)):
        eigval1, eigval2 = eigenvalues[i]
        eigvec1, eigvec2 = eigenvectors[i][:, 0], eigenvectors[i][:, 1]
        
        # Plot first eigenvector
        ax.quiver(coord_1[i], coord_2[i], eigvec1[0], eigvec1[1], 
                  angles='xy', scale_units='xy', scale=1/eigval1, color='r')
        
        # Plot second eigenvector
        ax.quiver(coord_1[i], coord_2[i], eigvec2[0], eigvec2[1], 
                  angles='xy', scale_units='xy', scale=1/eigval2, color='b')

# Function to plot the mesh and the eigenvectors
def plot_tensor_eigenvectors(mesh, scale=0.1):
    coord_1 = mesh.point_data['coord_1']
    coord_2 = mesh.point_data['coord_2']
    faces = mesh.faces.reshape((-1, 4))[:, 1:4]
    triangulation = tri.Triangulation(coord_1, coord_2, faces)
    
    # Get the curvature_tensor_avg data and reshape it into 2x2 tensors
    curvature_tensor_data = mesh.point_data['curvature_tensor_avg']
    eigenvalues, eigenvectors = calculate_eigenvectors(curvature_tensor_data)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the mesh using triangulation
    ax.tripcolor(triangulation, np.zeros_like(coord_1), shading='flat', cmap='Greys', alpha=0.3)
    
    # Plot eigenvectors on the mesh
    plot_eigenvectors(ax, mesh, eigenvalues, eigenvectors, scale=scale)
    
    ax.set_xlabel('Coord 1')
    ax.set_ylabel('Coord 2')
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()

# Function to plot eigenvectors for each mesh in the dataset, one after the other
def plot_all_eigenvectors(times, categories, meshes, scale=0.1):
    for i, mesh in enumerate(meshes):
        category, time = categories[i], times[i]
        print(f"Plotting for {category} at time {time}")
        plot_tensor_eigenvectors(mesh, scale=scale)