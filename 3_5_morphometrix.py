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

def count_line_occurrences(lines, faces):
    # Normalize line representation (ensure smaller vertex index comes first)
    sorted_lines = np.sort(lines, axis=1)
    
    # Generate all edges from faces and normalize
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])
    sorted_edges = np.sort(edges, axis=1)

    # Efficient counting of each line's occurrences in edges
    # Expand dimensions and compare
    expanded_lines = sorted_lines[:, np.newaxis, :]
    matches = np.all(expanded_lines == sorted_edges, axis=2)

    # Sum matches across edges to get occurrence counts
    N_occure = matches.sum(axis=1)
    print(N_occure)
    return N_occure
    

def getMorphoDf(mesh:pv.PolyData,labels):
    boundary = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False,non_manifold_edges=False,clear_data=True)
    print(boundary.lines)
    #boundary.plot(color='red', line_width=2, show_edges=True)
    lines = boundary.lines.reshape((-1, 3))[:, 1:3]
    print(lines)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    print(faces)
    line_occurrences = count_line_occurrences(lines, faces)
    print("Line occurrences:")
    for line, count in line_occurrences.items():
        print(f"Line {line}: {count} times")
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
