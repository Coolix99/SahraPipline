import numpy as np
from scipy.spatial import cKDTree

def dist_from_start(path):
    differences = np.diff(path, axis=0)
    # Calculate the Euclidean distance between each consecutive pair of points
    distances_between_points = np.linalg.norm(differences, axis=1)
    # Create an array with cumulative distances from the start
    cumulative_distances = np.cumsum(distances_between_points)
    # Prepend a 0 to represent the start
    distances_from_start = np.insert(cumulative_distances, 0, 0)

    return distances_from_start

def closest_point_on_segment(A, B, P):
    AB = B - A
    AP = P - A
    dot_product = np.sum(AP * AB, axis=1)
    denominator = np.sum(AB * AB, axis=1)
    factor = dot_product / denominator
    # Check if the factor is outside the [0, 1] range
    outside_segment = (factor < 0) | (factor > 1)
    # Set closest point to NaN if outside the segment
    closest_point = np.where(outside_segment[:, np.newaxis], np.inf, A + factor[:, np.newaxis] * AB)
    # Calculate distance between P and closest point X
    distance = np.linalg.norm(closest_point - P, axis=1)
    
    return closest_point, distance, factor

def closest_point_on_path(path,P):
    # Calculate the closest points and distances for each segment
    A_values = path[:-1]  # Starting points of segments
    B_values = path[1:]   # Ending points of segments
    closest_points, distances,_ = closest_point_on_segment(A_values, B_values, P)
    # Find the index of the closest point with the minimal distance
    min_distance_index = np.argmin(distances)
   
    # Return the closest point with the minimal distance
    closest_point = closest_points[min_distance_index]
    min_distance = distances[min_distance_index]

    tree = cKDTree(path)
    min_distance_point, index_point = tree.query(P, k=1)
    if min_distance_point<min_distance:
        closest_point=path[index_point]
        min_distance=min_distance_point

    return closest_point, min_distance    

def closest_position_on_path(path,P):
    # Calculate the closest points and distances for each segment
    A_values = path[:-1]  # Starting points of segments
    B_values = path[1:]   # Ending points of segments
    _, distances, factors = closest_point_on_segment(A_values, B_values, P)
    # Find the index of the closest point with the minimal distance
    min_distance_index = np.argmin(distances)
   
    # Return the closest point with the minimal distance
    min_distance = distances[min_distance_index]
    min_factor = factors[min_distance_index]

    tree = cKDTree(path)
    min_distance_point, index_point = tree.query(P, k=1)
    if min_distance_point<min_distance:
        return dist_from_start(path[:index_point+1,:])[-1]
    
    x=dist_from_start(path[:min_distance_index+1,:])[-1]
    x=x+min_factor*np.linalg.norm(A_values[min_distance_index,:]-B_values[min_distance_index,:])

    return x

def interpolate_on_path(path, pos):
    # Calculate the distances from the start
    distances_from_start = dist_from_start(path)
    
    # Find the indices of the segments that contain the positions
    segment_indices = np.searchsorted(distances_from_start, pos, side='right') - 1
    segment_indices=np.minimum(segment_indices, path.shape[0]-2)

    # Compute the relative positions within the segments
    segment_distances = pos - distances_from_start[segment_indices]
    segment_lengths = distances_from_start[segment_indices + 1] - distances_from_start[segment_indices]
    segment_ratios = segment_distances / segment_lengths
    # Interpolate points
    interpolated_points = path[segment_indices] + segment_ratios[:, np.newaxis] * (path[segment_indices + 1] - path[segment_indices])

    return interpolated_points
