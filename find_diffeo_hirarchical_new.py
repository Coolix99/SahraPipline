from typing import List,Tuple
import pyvista as pv
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit

from sub_manifold import sub_manifold,sub_manifold_1_closed,sub_manifold_0
from boundary import getBoundary,path_surrounds_point

def interpolateDeformation_localCoord(mesh:pv.PolyData): 
    mesh.point_data["displacement"]=mesh.point_data["deformed"]-mesh.point_data["rotated"]
    data_array = mesh.point_data["displacement"]
   
    nan_points = np.isnan(data_array).any(axis=1)
    #selected_vertices=np.where(~nan_points)[0]

    given_AP=mesh.point_data['AP mum'][~nan_points]
    given_PD=mesh.point_data['PD mum'][~nan_points]
    given_points=np.column_stack((given_AP, given_PD))
    nan_AP=mesh.point_data['AP mum'][nan_points]
    nan_PD=mesh.point_data['PD mum'][nan_points]
    wanted_points=np.column_stack((nan_AP, nan_PD))
    for i in range(3):
        given_values=data_array[~nan_points][:,i]
        data_array[nan_points, i]=griddata(given_points, given_values, wanted_points, method='linear')

    #fill holes
    from scipy.spatial import cKDTree
    nan_points = np.isnan(data_array).any(axis=1)
    non_nan_points =mesh.point_data["rotated"][~nan_points]
    tree = cKDTree(non_nan_points)
    # Loop over NaN points
    for i in np.where(nan_points)[0]:
        # Find the indices and distances to the nearest non-NaN points using Dijkstra's algorithm
        distances, indices = tree.query(mesh.point_data["rotated"][i], k=3)
        # Map the indices back to the original indices in mesh.point_data["rotated"]
        original_indices = np.where(~nan_points)[0][indices]
        weight = np.exp(-distances/2)
        interpolated_value = np.sum(data_array[original_indices, :] * weight[:, np.newaxis], axis=0) / np.sum(weight)

        data_array[i, :] = interpolated_value


    mesh.point_data["displacement"] = data_array
    mesh.point_data["deformed"] = mesh.point_data["rotated"] + mesh.point_data["displacement"]

def smoothDisplacement(mesh:pv.PolyData):
    mesh.point_data["displacement"]=mesh.point_data["deformed"]-mesh.point_data["rotated"]
    data_array = mesh.point_data["displacement"]

    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.point_data["rotated"])

    for i in range(mesh.point_data["rotated"].shape[0]):
        # Find the indices and distances to the nearest non-NaN points using Dijkstra's algorithm
        distances, indices = tree.query(mesh.point_data["rotated"][i], k=20)
        weight = np.exp(-distances/4)
        interpolated_value = np.sum(data_array[indices, :] * weight[:, np.newaxis], axis=0) / np.sum(weight)

        data_array[i, :] = interpolated_value

    # Assign the updated data_array back to the mesh
    mesh.point_data["displacement"] = data_array
    mesh.point_data["deformed"] = mesh.point_data["rotated"] + mesh.point_data["displacement"]

def projectPoints(mesh_init,mesh_target:pv.PolyData,sub_manifolds_init:List[sub_manifold],sub_manifolds_target:List[sub_manifold]):
    closest_cells, closest_points = mesh_target.find_closest_cell(mesh_init.point_data["deformed"], return_closest_point=True)
    mesh_init.point_data["deformed"]=closest_points
    
    for sub_init in sub_manifolds_init:
        for sub_target in sub_manifolds_target:
            if sub_init.name==sub_target.name:
                sub_init.project(sub_target)
                continue
    
    mesh_init.point_data["displacement"]=mesh_init.point_data["deformed"]-mesh_init.point_data["rotated"]

def findInitialGuess(mesh_init,mesh_target,sub_manifolds_init:List[sub_manifold],sub_manifolds_target:List[sub_manifold]):

    for sub_init in sub_manifolds_init:
        for sub_target in sub_manifolds_target:
            if sub_init.name==sub_target.name:
                sub_init.Initial_guess(sub_target)
                continue
    #interpolateDeformation(mesh_init)
    interpolateDeformation_localCoord(mesh_init)
    
    #project points
    projectPoints(mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target)
    smoothDisplacement(mesh_init)
    projectPoints(mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target)

def getTangentComponent(derivatives,mesh_init:pv.PolyData,mesh_target:pv.PolyData,sub_manifolds_init:List[sub_manifold]):
    closest_cells = mesh_target.find_closest_cell(mesh_init.point_data["deformed"])
    
    normal_vectors = mesh_target.cell_normals
    selected_vectors=normal_vectors[closest_cells,:]
    
    dot_product = np.sum(derivatives * selected_vectors, axis=1)
    projection = dot_product[:, np.newaxis] * selected_vectors
    tangent_derivatives = derivatives - projection

    for sub_init in sub_manifolds_init:
        tangent_derivatives=sub_init.getTangentDeriv(tangent_derivatives)

    return tangent_derivatives

def makeHirarchicalGrids(mesh_init:pv.PolyData,sub_manifolds_init:List[sub_manifold],mesh_target,target_reductions)-> Tuple[List, List]:
    #create decimated meshes
    mesh_hirarchical:List[pv.PolyData]=[]
    
    for target_reduction in target_reductions:
        mesh_hirarchical.append(mesh_init.decimate(target_reduction,volume_preservation=True))
    mesh_hirarchical.append(mesh_init)

    #add point_data
    point_data1 = mesh_init.point_data
    for i in range(len(mesh_hirarchical)-1):
        points2 = mesh_hirarchical[i].points
        closest_indices = []
        for point in points2:
            closest_index = mesh_init.find_closest_point(point)
            closest_indices.append(closest_index)
        # Transfer point data from mesh1 to mesh2 using closest indices
        for array_name, array in point_data1.items():
            mesh_hirarchical[i].point_data[array_name] = array[closest_indices]

    #adapt submanifolds
    sub_manifolds_hirarchical:List[List[sub_manifold]]=[]
    for i in range(len(mesh_hirarchical)-1):
        indices_init = getBoundary(mesh_hirarchical[i])
        if path_surrounds_point(mesh_hirarchical[i].point_data['rotated'][indices_init],mesh_target.center,mesh_target.n_target)=='Counterclockwise':
            indices_init=indices_init[::-1]
        sub_manifolds:List[sub_manifold]=[sub_manifold_1_closed(mesh_hirarchical[i], 'boundary', indices_init)]
        for sub_init in sub_manifolds_init: 
            if sub_init.name=='boundary':
                continue
            sub_manifolds.append(sub_init.getTransferedSMf(mesh_hirarchical[i]))       
        sub_manifolds_hirarchical.append(sub_manifolds)
    sub_manifolds_hirarchical.append(sub_manifolds_init)
    

    return mesh_hirarchical,sub_manifolds_hirarchical

def transfer_displacement(mesh_coarse ,mesh_fine):
    data_array = np.full((mesh_fine.point_data['AP mum'].shape[0], 3), np.nan)
   
    given_AP=mesh_coarse.point_data['AP mum']
    given_PD=mesh_coarse.point_data['PD mum']
    given_points=np.column_stack((given_AP, given_PD))
    wanted_AP=mesh_fine.point_data['AP mum']
    wanted_PD=mesh_fine.point_data['PD mum']
    wanted_points=np.column_stack((wanted_AP, wanted_PD))
    for i in range(3):
        given_values=mesh_coarse.point_data["displacement"][:,i]
        data_array[:, i]=griddata(given_points, given_values, wanted_points, method='linear')

    #fill holes
    from scipy.spatial import cKDTree
    nan_points = np.isnan(data_array).any(axis=1)
    non_nan_points =mesh_fine.point_data["rotated"][~nan_points]
    tree = cKDTree(non_nan_points)
    # Loop over NaN points
    for i in np.where(nan_points)[0]:
        # Find the indices and distances to the nearest non-NaN points using Dijkstra's algorithm
        distances, indices = tree.query(mesh_fine.point_data["rotated"][i], k=3)
        # Map the indices back to the original indices in mesh.point_data["rotated"]
        original_indices = np.where(~nan_points)[0][indices]
        weight = np.exp(-distances/2)
        interpolated_value = np.sum(data_array[original_indices, :] * weight[:, np.newaxis], axis=0) / np.sum(weight)

        data_array[i, :] = interpolated_value

    mesh_fine.point_data["displacement"] = data_array
    mesh_fine.point_data["deformed"] = mesh_fine.point_data["rotated"] + mesh_fine.point_data["displacement"]

def compute_deformation_gradient_vectorized(X, x):
    """
    Compute the deformation gradient F for each face.
    X and x are of shape (N_f, 3, 3), where N_f is the number of faces.
    """
    # Edge vectors in the reference and deformed configuration (N_f x 3 x 3)
    DX1 = X[:, 1] - X[:, 0]  # First edge
    DX2 = X[:, 2] - X[:, 0]  # Second edge

    dx1 = x[:, 1] - x[:, 0]  # Deformed first edge
    dx2 = x[:, 2] - x[:, 0]  # Deformed second edge

    # Compute normal vector for the reference configuration
    N_ref = jnp.cross(DX1, DX2)  # Cross product gives normal to the triangle
    N_ref = N_ref / jnp.linalg.norm(N_ref, axis=-1, keepdims=True)  # Normalize

    # Compute normal vector for the deformed configuration
    N_def = jnp.cross(dx1, dx2)  # Cross product gives deformed normal
    N_def = N_def / jnp.linalg.norm(N_def, axis=-1, keepdims=True)  # Normalize

    # Construct the 3x3 deformation gradient matrix
    DX_full = jnp.stack([DX1, DX2, N_ref], axis=-1)  # Shape: (N_f, 3, 3)
    dx_full = jnp.stack([dx1, dx2, N_def], axis=-1)  # Shape: (N_f, 3, 3)

    # Solve the system DX_full @ F.T = dx_full to get the deformation gradient F
    F = jnp.einsum('...ij,...jk->...ik', dx_full, jnp.linalg.pinv(DX_full))

    return F

def mooney_rivlin_energy(F, a, b, c, d):
    """
    Compute the corrected Mooney–Rivlin energy for each face.
    """
    
    # First term: a * (||F||^2 - 3)
    term1 = a * (jnp.sum(F**2, axis=(-2, -1)) - 3)

    # Second term: b * (||Cof(F)||^2 - 3)
    Cof_F = jnp.linalg.det(F)[:, None, None] * jnp.linalg.inv(F).transpose(0, 2, 1)
    term2 = b * (jnp.sum(Cof_F**2, axis=(-2, -1)) - 3)
    
    # Third term: Γ(det(F)) = c * (det(F)^2 - 1) - d * log(det(F))
    det_F = jnp.linalg.det(F)
    term3 = c * (det_F*det_F - 1) - d * jnp.log(det_F)

    # Total energy
    W = term1 + term2 + term3

    return W

def compute_triangle_areas(X):
    """
    Compute the area of each triangle given the vertices.
    X is of shape (N_f, 3, 3).
    """
    # Edge vectors
    DX1 = X[:, 1] - X[:, 0]
    DX2 = X[:, 2] - X[:, 0]

    # Cross product gives a vector with magnitude equal to twice the area of the triangle
    cross_product = jnp.cross(DX1, DX2)
    areas = 0.5 * jnp.linalg.norm(cross_product, axis=-1)

    return areas

@jit
def total_energy(x, X, a, b, c,d):
    """
    Compute the total energy by integrating the energy over all faces using their areas.
    """
    # Compute deformation gradient F for each face
    F = compute_deformation_gradient_vectorized(X, x)

    # Compute the Mooney-Rivlin energy for each face
    W = mooney_rivlin_energy(F, a, b, c,d)
  
    # Compute the areas of the triangles in the reference configuration
    areas = compute_triangle_areas(X)
    
    # Integrate the energy over the triangles by multiplying the energy by the areas
    total_W = jnp.sum(W * areas)

    return total_W

@jit
def compute_forces(vertex_points, deformed_vertex_points, faces, a, b, c,d):
    # Gather the original and deformed positions for each face
    X = vertex_points[faces]  # Shape: (N_f, 3, 3)
    x = deformed_vertex_points[faces]  # Shape: (N_f, 3, 3)

    # Gradient of total energy with respect to the deformed vertex positions
    dW_x = grad(total_energy, argnums=0)(x, X, a, b, c,d)

    # Initialize forces with zeros
    forces = jnp.zeros_like(deformed_vertex_points)

    # Distribute forces back to vertices
    for i in range(3):
        forces = -forces.at[faces[:, i]].add(dW_x[:, i])

    return forces

def total_energy_from_vertices(vertex_points, deformed_vertex_points, faces, a, b, c,d):
    X = vertex_points[faces] 
    x = deformed_vertex_points[faces]  
    return np.array(total_energy(x, X, a, b, c,d))


def findDiffeo(mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target):
    N_iterations=[20,20,20,20]
    start_rates=[0.2,0.2,0.05,0.025]
    target_reductions=[0.95,0.7,0.3]
    mesh_hirarchical,sub_manifolds_hirarchical=makeHirarchicalGrids(mesh_init,sub_manifolds_init,mesh_target,target_reductions)
    # p = pv.Plotter()
    # delta=0
    # for i in range(len(mesh_hirarchical)):
    #     mesh_hirarchical[i].points=mesh_hirarchical[i].point_data['rotated']-np.array((delta,0,0))
    #     p.add_mesh(mesh_hirarchical[i], color='blue', show_edges=True)
    #     points_array=mesh_hirarchical[i].point_data['rotated'][sub_manifolds_hirarchical[i][0].path]-np.array((delta,0,0))
    #     polydata = pv.PolyData(points_array)
    #     lines=np.zeros((len(points_array)+1),dtype=int)
    #     lines[0]=len(points_array)
    #     lines[1:] = np.arange(len(points_array),dtype=int)
    #     polydata.lines = lines
    #     p.add_mesh(polydata, color='blue', line_width=5, render_lines_as_tubes=True)  
    #     delta+=50
    # p.show()
    
    findInitialGuess(mesh_hirarchical[0],mesh_target,sub_manifolds_hirarchical[0],sub_manifolds_target) #for first mesh
    # for valid energy: (2a+6b+2c−d)=0
    a=0.1
    b=0.1
    c=1.0  
    d=2*a+4*b+2*c
    for level in range(len(mesh_hirarchical)):
        all_sum_deriv=[]
        #interative solving
        rate=start_rates[level]
        faces = mesh_hirarchical[level].faces.reshape((-1, 4))[:, 1:4]
        vertex_points=mesh_hirarchical[level].points
        for i in range(N_iterations[level]):
            #print(i)
            #calculate the forces
            
            deformed_vertex_points=mesh_hirarchical[level].point_data['deformed']
            forces=np.array(compute_forces(vertex_points, deformed_vertex_points, faces, a, b, c,d))

            #calculate normal component or take paralel component
            tangent_forces=getTangentComponent(forces,mesh_hirarchical[level],mesh_target,sub_manifolds_hirarchical[level])

            #do euler step
            max_force=np.max(tangent_forces)
            #print(max_deriv)
            sum_forces=np.sum(np.linalg.norm(tangent_forces,axis=1))
            #print(sum_deriv)
            all_sum_deriv.append(sum_forces)
            if i>2:
                if all_sum_deriv[-2]<all_sum_deriv[-1]:
                    print(rate)
                    rate=rate/1.5

            mesh_hirarchical[level].point_data["deformed"]=mesh_hirarchical[level].point_data["deformed"]+rate*tangent_forces/max_force
            projectPoints(mesh_hirarchical[0],mesh_target,sub_manifolds_hirarchical[level],sub_manifolds_target)
            mesh_hirarchical[level].point_data["displacement"]=mesh_hirarchical[level].point_data["deformed"]-mesh_hirarchical[level].point_data["rotated"]
        #transfer result to next level
        if level<len(mesh_hirarchical)-1:
            transfer_displacement(mesh_hirarchical[level],mesh_hirarchical[level+1])   
            smoothDisplacement(mesh_hirarchical[level+1])
            projectPoints(mesh_hirarchical[level+1],mesh_target,sub_manifolds_hirarchical[level+1],sub_manifolds_target)

        # plt.plot(range(len(all_sum_deriv)),all_sum_deriv)
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.show()

        if level==len(mesh_hirarchical)-1:
            return total_energy_from_vertices(vertex_points,deformed_vertex_points,faces, a, b, c,d)
       
    

     


def main():
    def find_transformation_matrix(v1, v2, v3, u1, u2, u3):
        """Find the transformation matrix that maps v vectors to u vectors."""
        # Concatenate the v and u vectors into a single matrix
        A = np.vstack((v1, v2, v3)).T
        B = np.vstack((u1, u2, u3)).T
        transformation_matrix=B@np.linalg.inv(A)
        return transformation_matrix

    def transformation_matrix(v1, v2, v3, u1, u2, u3, center_point1, center_point2):
        """Find the transformation matrix that maps unit_vector1 to unit_vector2
        and center_point1 to center_point2."""
        rotation_matrix = find_transformation_matrix(v1, v2, v3, u1, u2, u3)
        translation_vector = center_point2 - np.dot(rotation_matrix, center_point1)

        return rotation_matrix,translation_vector

    def extract_coordinate(df, name):
        row = df[df['name'] == name]
        if not row.empty:
            return row['coordinate_px'].iloc[0]
        else:
            return None


    def getInput(init_dir_path,target_dir_path):
        from IO import get_JSON
        import pandas as pd
        init_MetaData=get_JSON(init_dir_path)
        try:
            init_surface_file_name=init_MetaData['Thickness_MetaData']['Surface file']
            init_orient_file_name=init_MetaData['Orient_MetaData']['Orient file']
        except:
            return False
        target_MetaData=get_JSON(target_dir_path)
        try:
            target_surface_file_name=target_MetaData['Thickness_MetaData']['Surface file']
            target_orient_file_name=target_MetaData['Orient_MetaData']['Orient file']
        except:
            return False
        
        scales_init=init_MetaData['CenterLine_MetaData']['scales ZYX'].copy()
        scales_target=target_MetaData['CenterLine_MetaData']['scales ZYX'].copy()
    
        init_surface_file=os.path.join(init_dir_path,init_surface_file_name)
        init_orient_file=os.path.join(init_dir_path,init_orient_file_name)
        target_surface_file=os.path.join(target_dir_path,target_surface_file_name)
        target_orient_file=os.path.join(target_dir_path,target_orient_file_name)

        df_init_orient = pd.read_hdf(init_orient_file)
        df_target_orient = pd.read_hdf(target_orient_file)
        mesh_init = pv.read(init_surface_file)
        mesh_target = pv.read(target_surface_file)

        #print(df_init_orient)
        #print(df_target_orient)
        # print(mesh_init)
        # print(mesh_target)
        v1_init=(extract_coordinate(df_init_orient,'Proximal_pt')-extract_coordinate(df_init_orient,'Distal_pt'))*scales_init
        v2_init=(extract_coordinate(df_init_orient,'Anterior_pt')-extract_coordinate(df_init_orient,'Posterior_pt'))*scales_init
        v3_init=(extract_coordinate(df_init_orient,'Proximal2_pt')-extract_coordinate(df_init_orient,'Distal2_pt'))*scales_init
        n_init=np.cross(v3_init,v2_init)
        n_init = n_init / np.linalg.norm(n_init)
        v1_init = v1_init - n_init*np.dot(n_init,v1_init)
        v1_init = v1_init / np.linalg.norm(v1_init)
        v4_init = np.cross(n_init,v1_init)
        
        v1_target=(extract_coordinate(df_target_orient,'Proximal_pt')-extract_coordinate(df_target_orient,'Distal_pt'))*scales_target
        v2_target=(extract_coordinate(df_target_orient,'Anterior_pt')-extract_coordinate(df_target_orient,'Posterior_pt'))*scales_target
        v3_target=(extract_coordinate(df_target_orient,'Proximal2_pt')-extract_coordinate(df_target_orient,'Distal2_pt'))*scales_target
        n_target=np.cross(v3_target,v2_target)
        n_target = n_target / np.linalg.norm(n_target)
        v1_target = v1_target - n_target*np.dot(n_target,v1_target)
        v1_target = v1_target / np.linalg.norm(v1_target)
        v4_target = np.cross(n_target,v1_target)
        mesh_target.n_target=n_target

        center_init=mesh_init.center
        center_target=mesh_target.center

        rotation_matrix,translation_vector = transformation_matrix(v1_init,n_init,v4_init, v1_target,n_target,v4_target, center_init, center_target)
        # print(rotation_matrix@center_init+translation_vector)
        # print(rotation_matrix@v1_init)
        # print(rotation_matrix@n_init)
        # print(rotation_matrix@v4_init)
        mesh_init.point_data['rotated']=(rotation_matrix@mesh_init.points.T+translation_vector[:,np.newaxis]).T
    
        #calculate boundary
        #make submanifolds
        indices_init = getBoundary(mesh_init)
        if path_surrounds_point(mesh_init.point_data['rotated'][indices_init],center_target,n_target)=='Counterclockwise':
            indices_init=indices_init[::-1]
        sub_manifolds_init:List[sub_manifold]=[sub_manifold_1_closed(mesh_init, 'boundary', indices_init)]

        indices_target = getBoundary(mesh_target)
        if path_surrounds_point(mesh_target.points[indices_target],center_target,n_target)=='Counterclockwise':
            indices_target=indices_target[::-1]
        sub_manifolds_target:List[sub_manifold]=[sub_manifold_1_closed(mesh_target, 'boundary', indices_target)]

        return mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target,init_MetaData,target_MetaData

    import os
    from config import FlatFin_path
    init_dir_path=os.path.join(FlatFin_path,'270524_sox10_claudin-gfp_72h_pecfin2_FlatFin')
    target_dir_path=os.path.join(FlatFin_path,'270524_sox10_claudin-gfp_72h_pecfin5_FlatFin')
    mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target,init_MetaData,target_MetaData=getInput(init_dir_path,target_dir_path)

    findDiffeo(mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target)

    p = pv.Plotter()
    p.add_mesh(mesh_target, color='green', show_edges=True)
    points_array=mesh_target.points[sub_manifolds_target[0].path]
    polydata = pv.PolyData(points_array)
    lines=np.zeros((len(points_array)+1),dtype=int)
    lines[0]=len(points_array)
    lines[1:] = np.arange(len(points_array),dtype=int)
    polydata.lines = lines
    p.add_mesh(polydata, color='green', line_width=5, render_lines_as_tubes=True)  
    delta=30
    mesh_init.points=mesh_init.point_data['deformed']-np.array((delta,0,0))
    p.add_mesh(mesh_init, color='blue', show_edges=True)
    points_array=mesh_init.point_data['deformed'][sub_manifolds_init[0].path]-np.array((delta,0,0))
    polydata = pv.PolyData(points_array)
    lines=np.zeros((len(points_array)+1),dtype=int)
    lines[0]=len(points_array)
    lines[1:] = np.arange(len(points_array),dtype=int)
    polydata.lines = lines
    p.add_mesh(polydata, color='blue', line_width=5, render_lines_as_tubes=True)  
   
    p.show()


    return


if __name__ == "__main__":
    main()