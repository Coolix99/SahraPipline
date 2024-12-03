import numpy as np
from skimage.graph import pixel_graph
from scipy.sparse import csr_matrix

def different_labels(center_value, neighbor_value, *args):
    """Return 1.0 when two labels are different, otherwise 0.0."""
    return (center_value != neighbor_value).astype(float)

def find_touching_pairs(label_mask):
    """
    Find unique pairs of touching objects in a 3D segmented image.
    
    Parameters:
        label_mask (np.ndarray): 3D array with integer labels.
        
    Returns:
        list[tuple]: List of unique sorted pairs of touching objects.
    """
    # Ensure the input is boolean to define valid regions
    mask = label_mask > 0

    # Generate the pixel graph
    g, nodes = pixel_graph(
        label_mask,
        mask=mask,
        connectivity=3,  # Use full 3D connectivity
        edge_function=different_labels
    )

    # Remove zero entries from the graph
    g.eliminate_zeros()

    # Convert to COO format for easy manipulation
    coo = g.tocoo()

    # Get the pixel indices and their labels
    center_coords = nodes[coo.row]
    neighbor_coords = nodes[coo.col]

    center_values = label_mask.ravel()[center_coords]
    neighbor_values = label_mask.ravel()[neighbor_coords]

    # Create a set of unique pairs
    pairs = set()
    for i, j in zip(center_values, neighbor_values):
        if i != 0 and j != 0 and i != j:  # Exclude background and self-relations
            pairs.add(tuple(sorted((i, j))))  # Sort the tuple to ensure uniqueness

    # Convert the set to a list of tuples
    return list(pairs)

# Example Usage
if __name__ == "__main__":
    # Create a dummy 3D segmented array
    label_mask = np.zeros((5, 5, 5), dtype=int)
    label_mask[1:3, 1:3, 1:3] = 1  # Object 1
    label_mask[2:4, 2:4, 2:4] = 2  # Object 2
    label_mask[3:5, 3:5, 3:5] = 3  # Object 3
    print(label_mask)
    # Get touching pairs
    touching_pairs = find_touching_pairs(label_mask)

    # Print results
    print("Touching pairs:")
    for pair in touching_pairs:
        print(pair)
