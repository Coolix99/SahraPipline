import pyvista as pv
import numpy as np
import pymeshfix as mf

def sort_ind(connections):
    sorted_array = np.sort(connections, axis=1)

    tuple_set = {tuple(row) for row in sorted_array}
    first=tuple_set.pop()
    sorted_indices=[first[0],first[1]]

    while len(tuple_set)>0:
        for connection in tuple_set:
            found=0
            if connection[0]==sorted_indices[-1]:
                sorted_indices.append(connection[1])
                tuple_set.discard(connection)
                found=1
                break
            if connection[1]==sorted_indices[-1]:
                sorted_indices.append(connection[0])
                tuple_set.discard(connection)
                found=1
                break
        if(found==0):
            break
    return sorted_indices

def getBoundary(surf:pv.PolyData):
    meshfix = mf.MeshFix(surf.triangulate())
    holes = meshfix.extract_holes()
    # indices1 = np.arange(len(holes.lines)) % 3 == 1
    # indices2 = np.arange(len(holes.lines)) % 3 == 2
    # v1=holes.lines[indices1]
    # v2=holes.lines[indices2]
    # result=np.zeros((v1.shape[0],2),dtype=np.uint)
    # result[:,0] = v1
    # result[:,1] = v2

    boundary_points = holes.points
    boundary_vertex_indices = np.empty(boundary_points.shape[0], dtype=int)
    for i, point in enumerate(boundary_points):
        boundary_vertex_indices[i] = surf.find_closest_point(point)

    simplices=surf.faces.reshape(-1, 4)[:, 1:]
    filtered_indices = []
    tips = []

    for triangle in simplices:
        triangle_indices = [idx for idx in triangle if idx in boundary_vertex_indices]
        if len(triangle_indices) == 2:
            filtered_indices.append(triangle_indices)
        if len(triangle_indices) == 3:
            tips.append(triangle_indices)

    filtered_simplices = np.array(filtered_indices)

    sorted_indices=sort_ind(filtered_simplices)
    for t in tips:
        if t[1] in sorted_indices and t[2] in sorted_indices:
            index_b = sorted_indices.index(t[1])
            index_c = sorted_indices.index(t[2])
            sorted_indices.insert(min(index_b, index_c)+1, t[0])
            continue
        if t[0] in sorted_indices and t[2] in sorted_indices:
            index_a = sorted_indices.index(t[0])
            index_c = sorted_indices.index(t[2])
            sorted_indices.insert(min(index_a, index_c)+1, t[1])
            continue
        if t[0] in sorted_indices and t[1] in sorted_indices:
            index_a = sorted_indices.index(t[0])
            index_b = sorted_indices.index(t[1])
            sorted_indices.insert(min(index_a, index_b)+1, t[2])
            continue

    # print("sorted:")
    # print(sorted_indices)

    return sorted_indices[0:-1]

def path_surrounds_point(path_array, p, n):
    # Calculate vectors from each point to the next
    vectors = np.diff(path_array, axis=0)
    # Calculate cross products with n
    cross_products = np.cross(vectors, n)
    

    # Vector from p to the first point in the path
    vector_to_first_point = path_array - p

    # Dot product between summed vector and vector_to_first_point
    dot_product =np.sum(cross_products * vector_to_first_point[:-1], axis=(1,0))
    if dot_product > 0:
        return "Clockwise"
    elif dot_product < 0:
        return "Counterclockwise"
    else:
        return "Indeterminate"
