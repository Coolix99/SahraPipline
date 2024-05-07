import numpy as np
#np.random.seed(42)
from typing import List
import napari
import os
import git
from simple_file_checksum import get_checksum
import pyvista as pv



from config import *
from IO import *
from boundary import getBoundary

def getBoundary(faces):
    # Create all edges by pairing vertices
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])
    
    # Sort edges to normalize direction
    sorted_edges = np.sort(edges, axis=1)
    
    # Find unique edges and their counts
    unique_edges, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    print(sorted_edges.shape)
    print(unique_edges.shape)
    print(counts.shape)
    # Filter edges that appear exactly once
    boundary_edges = unique_edges[counts == 1]
    
    return boundary_edges


def filter_edges(edges):
    # Count occurrences of each vertex in the first and second positions
    first_vertices, first_counts = np.unique(edges[:, 0], return_counts=True)
    second_vertices, second_counts = np.unique(edges[:, 1], return_counts=True)

    # Determine which vertices appear at least twice in each position
    valid_first = first_vertices[first_counts >= 2]
    valid_second = second_vertices[second_counts >= 2]

    # Create masks to filter edges
    valid_first_mask = np.isin(edges[:, 0], valid_first)
    valid_second_mask = np.isin(edges[:, 1], valid_second)

    # Filter edges where both vertices meet the appearance criterion
    valid_edges = edges[valid_first_mask & valid_second_mask]

    return valid_edges

def detect_non_conformal_vertices(mesh):
    # Extract the faces of the mesh
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    
    # Get boundary edges
    boundary_edges = getBoundary(faces)
    print(boundary_edges.shape)
    last=boundary_edges.shape[0]
    while True:
        boundary_edges=filter_edges(boundary_edges)
        new_length=boundary_edges.shape[0]
        if  last <= new_length:
            break
        last = new_length
    print(boundary_edges.shape)
    
    res=[]
    for i,edge in enumerate(boundary_edges):
        a=edge[0]
        b=edge[1]
        candidate_left=np.where(boundary_edges[:,0]==a)[0]
        candidate_right=np.where(boundary_edges[:,1]==b)[0]

        candidate_left = candidate_left[candidate_left != i]
        candidate_right = candidate_right[candidate_right != i]

        mid_left = boundary_edges[candidate_left,1]
        mid_right = boundary_edges[candidate_right,0]

        set1 = set(mid_left)
        set2 = set(mid_right)
        common_elements = set1.intersection(set2)
        common_elements_list = list(common_elements)
        if len(common_elements_list)==1:
            c=common_elements_list[0]
            res.append([a,b,c])
    res=np.array(res)
    print(res.shape)
    for i,(a, b, c) in enumerate(res):
        print(i)
        # Extract face information, assuming all triangles
        faces = mesh.faces.reshape(-1, 4)[:, 1:]

        # Get indices where vertices 'a' and 'b' appear
        face_indices_a = np.where((faces == a))[0]
        face_indices_b = np.where((faces == b))[0]
        
        # Determine the common face using set intersection
        common_faces = set(face_indices_a).intersection(face_indices_b)
        
        # Skip if there isn't exactly one common face
        if len(common_faces) != 1:
            continue
        
        # Get the common face ID
        face_id = list(common_faces)[0]
        face = faces[face_id]
        
        # Calculate the remaining vertex in the face
        remaining_vertex = np.setdiff1d(face, [a, b])[0]
        
        # Define the new faces resulting from the bisection
        new_faces = np.array([
            [3, a, c, remaining_vertex],
            [3, c, b, remaining_vertex]
        ])
        
        # Replace the original face in the 'mesh.faces' array
        original_faces = mesh.faces.reshape(-1, 4).copy()

        original_faces[face_id, :] = new_faces[0]
        # Append the second new face to the end of the 'mesh.faces' array
        mesh.faces = np.hstack([original_faces.flatten(), new_faces[1]])

    
    
    mesh = pv.PolyData(mesh.points, np.hstack(mesh.faces))
    mesh.plot(show_edges=True)
    return

def getMorphoDf(mesh:pv.PolyData, labels):
    print(mesh.points.shape)
    print(mesh.point_normals.shape)
    #boundary = mesh.extract_feature_edges(boundary_edges=False, feature_edges=True, manifold_edges=False,non_manifold_edges=False,clear_data=True)
    #boundary.plot(color='red', line_width=2, show_edges=True)
    

    return
    non_conformal_vertices = detect_non_conformal_vertices(mesh)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', show_edges=True)
    plotter.add_mesh(mesh.points[non_conformal_vertices], color='red', point_size=10, render_points_as_spheres=True, label='Non-Conformal Vertices')
    plotter.add_legend()
    plotter.show()
    

    return


    points = mesh.points
    lines = np.hstack((np.full((len(boundary), 1), 2), boundary)).astype(np.int64)

    # Convert these lines into a vtkLines object
    edge_mesh = pv.PolyData(points)
    edge_mesh.lines = lines

    # Create a plotter, add the mesh and the boundary edges
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', show_edges=True, label='Mesh')
    plotter.add_mesh(edge_mesh, color='red', line_width=3, label='Boundary Edges')

    # Optional: Add labels
    plotter.add_legend()

    # Display the plot
    plotter.show()
    return
    

    return
    boundary = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,non_manifold_edges=False,clear_data=True)
    boundary.plot(color='red', line_width=2, show_edges=True)
    return
    unique_labels = np.unique(labels)
    for label in unique_labels:
        print(label)
        selected_indices = np.where(labels == label)[0]
        selected_vertices = mesh.points[selected_indices]
        centroid = np.mean(selected_vertices, axis=0)

        u, s, vh = np.linalg.svd(selected_vertices - centroid)
        normal = vh[2, :]
        a, b, c = normal
        d = np.dot(normal, centroid)
        distances = (np.dot(selected_vertices, normal) - d) / (a**2 + b**2 + c**2)
        projections = selected_vertices - np.outer(distances, normal)

        u_proj, s_proj, vh_proj = np.linalg.svd(projections - centroid)
        basis1 = vh_proj[0]
        basis2 = vh_proj[1]

        centered_projections = projections - centroid
        coefficients = np.dot(centered_projections, np.vstack([basis1, basis2]).T)
        
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        mask = np.isin(faces, selected_indices).all(axis=1)
        filtered_faces = faces[mask]
        reverse_index_mapping = np.full(mesh.n_points, -1, dtype=int)  # Fill with -1 to indicate unused vertices
        reverse_index_mapping[selected_indices] = np.arange(len(selected_indices))  # Assign new indices
        relevant_faces = reverse_index_mapping[filtered_faces]

        if relevant_faces.shape[0]<1:
            continue
       
        fig, ax = plt.subplots()
        triangulation = mtri.Triangulation(coefficients[:, 0], coefficients[:, 1], triangles=relevant_faces)
        ax.triplot(triangulation, 'go-')
        ax.set_title('2D Projection of Selected Mesh with Connectivity')
        plt.show()


def evalStatus_morpho(fin_dir_path):
    MetaData=get_JSON(fin_dir_path)
    if not 'Mesh_MetaData' in MetaData:
        print('no Mesh_MetaData')
        return False
    
    if not 'project_MetaData' in MetaData:
        print('no project_MetaData')
        return False
    
    if not 'apply_MetaData' in MetaData:
        print('no apply_MetaData')
        return False
    
    if not 'seg_MetaData' in MetaData:
        print('no seg_MetaData')
        return False
    
    if not 'morpho_MetaData' in MetaData:
        return MetaData

    if not MetaData['morpho_MetaData']['morpho version']==morpho_version:
        return MetaData  

    if not MetaData['morpho_MetaData']['input seg checksum']==MetaData['seg_MetaData']['output seg checksum']:
        return MetaData

    return False

def morpho_analytic():  
    fin_list=[folder for folder in os.listdir(EpSeg_path) if os.path.isdir(os.path.join(EpSeg_path, folder))]
    for fin_folder in fin_list:
        print(fin_folder)
    
        fin_dir_path=os.path.join(EpSeg_path,fin_folder)
        
        PastMetaData=evalStatus_morpho(fin_dir_path)
        if not isinstance(PastMetaData,dict):
            continue

        MetaData_mesh=PastMetaData['Mesh_MetaData']
        Mesh_file=MetaData_mesh['Mesh file']
        Mesh_file_path=os.path.join(fin_dir_path,Mesh_file)
        mesh=pv.read(Mesh_file_path)

        MetaData_apply=PastMetaData['seg_MetaData']
        seg_file=MetaData_apply['seg file']
        seg_file_path=os.path.join(fin_dir_path,seg_file)
        labels=loadArr(seg_file_path)

        getMorphoDf(mesh,labels)

        

        return
        
        
        
        unique_integers = np.unique(labels)
        colors = plt.cm.get_cmap('nipy_spectral', len(unique_integers))
        np.random.seed(42) 
        indices = np.arange(len(unique_integers))
        np.random.shuffle(indices)
        color_mapping = {int_val: colors(idx / len(unique_integers)) for idx, int_val in zip(indices, unique_integers)}
        vertex_colors = np.array([color_mapping[val] for val in labels])
        viewer = napari.Viewer()
        vertices = mesh.points
        faces = mesh.faces.reshape((-1, 4))[:, 1:4]
        viewer.add_surface((vertices, faces),vertex_colors=vertex_colors, name='Surface Mesh')
        napari.run()
        return
        
        seg_file_name=fin_folder+'_seg'
        saveArr(labels,os.path.join(fin_dir_path,seg_file_name))
        
        MetaData_seg={}
        repo = git.Repo(gitPath,search_parent_directories=True)
        sha = repo.head.object.hexsha
        MetaData_seg['git hash']=sha
        MetaData_seg['git repo']='Sahrapipline'
        MetaData_seg['seg version']=seg_version
        MetaData_seg['seg file']=seg_file_name
        MetaData_seg['scales']=MetaData_mesh['scales']
        MetaData_seg['condition']=MetaData_mesh['condition']
        MetaData_seg['time in hpf']=MetaData_mesh['time in hpf']
        MetaData_seg['experimentalist']='Sahra'
        MetaData_seg['genotype']='WT'
        MetaData_seg['input flow checksum']=MetaData_apply['output flow checksum']
        check_proj=get_checksum(os.path.join(fin_dir_path,seg_file_name), algorithm="SHA1")
        MetaData_seg['output seg checksum']=check_proj
        writeJSON(fin_dir_path,'seg_MetaData',MetaData_seg)       


if __name__ == "__main__":
    morpho_analytic() 
