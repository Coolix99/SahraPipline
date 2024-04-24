from typing import List,Tuple
import pyvista as pv
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

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

def getA(triangle_vertices):
    x1=triangle_vertices[0,0]
    y1=triangle_vertices[0,1]
    z1=triangle_vertices[0,2]
    x2=triangle_vertices[1,0]
    y2=triangle_vertices[1,1]
    z2=triangle_vertices[1,2]
    x3=triangle_vertices[2,0]
    y3=triangle_vertices[2,1]
    z3=triangle_vertices[2,2]

    dx1=x2-x1
    dy1=y2-y1
    dz1=z2-z1

    dx2=x3-x1
    dy2=y3-y1
    dz2=z3-z1

    area=0.5*np.sqrt((dy1*dz2-dz1*dy2)*(dy1*dz2-dz1*dy2)
                        +(dz1*dx2-dx1*dz2)*(dz1*dx2-dx1*dz2)
                        +(dx1*dy2-dy1*dx2)*(dx1*dy2-dy1*dx2))
    if area<1e-4:
        area=1e-4
    return area

def getdA(triangle_vertices,A0):
    x1=triangle_vertices[0,0]
    y1=triangle_vertices[0,1]
    z1=triangle_vertices[0,2]
    x2=triangle_vertices[1,0]
    y2=triangle_vertices[1,1]
    z2=triangle_vertices[1,2]
    x3=triangle_vertices[2,0]
    y3=triangle_vertices[2,1]
    z3=triangle_vertices[2,2]

    dx1=x2-x1
    dy1=y2-y1
    dz1=z2-z1

    dx2=x3-x1
    dy2=y3-y1
    dz2=z3-z1

    dA1_x=1/A0 *(
        (dz1*dx2-dx1*dz2)*(-dz1+dz2)
        +(dx1*dy2-dy1*dx2)*(-dy2+dy1)
    )
    dA1_y=1/A0 *(
        (dy1*dz2-dz1*dy2)*(-dz2+dz1)
        +(dx1*dy2-dy1*dx2)*(-dx1+dx2)
    )
    dA1_z=1/A0 *(
        (dy1*dz2-dz1*dy2)*(-dy1+dy2)
        +(dz1*dx2-dx1*dz2)*(-dx2+dx1)
    )
    dA1=np.array((dA1_x,dA1_y,dA1_z))

    dA2_x=1/A0 *(
        (dz1*dx2-dx1*dz2)*(-dz2)
        +(dx1*dy2-dy1*dx2)*(dy2)
    )
    dA2_y=1/A0 *(
        (dy1*dz2-dz1*dy2)*(dz2)
        +(dx1*dy2-dy1*dx2)*(-dx2)
    )
    dA2_z=1/A0 *(
        (dy1*dz2-dz1*dy2)*(-dy2)
        +(dz1*dx2-dx1*dz2)*(dx2)
    )
    dA2=np.array((dA2_x,dA2_y,dA2_z))

    dA3_x=1/A0 *(
        (dz1*dx2-dx1*dz2)*(dz1)
        +(dx1*dy2-dy1*dx2)*(-dy1)
    )
    dA3_y=1/A0 *(
        (dy1*dz2-dz1*dy2)*(-dz1)
        +(dx1*dy2-dy1*dx2)*(dx1)
    )
    dA3_z=1/A0 *(
        (dy1*dz2-dz1*dy2)*(dy1)
        +(dz1*dx2-dx1*dz2)*(-dx1)
    )
    dA3=np.array((dA3_x,dA3_y,dA3_z))

    return dA1,dA2,dA3

def getPartialDerivatives(mesh_init:pv.PolyData):
    #E= int (A_init-A_def)^2/A_def^2 
    #A... area of triangle
    n_points = mesh_init.n_points

    derivatives=np.zeros((n_points,3))

    for i,cell in enumerate(mesh_init.cell):
        point_ids=cell.point_ids
        deformed_triangle_vertices = mesh_init.point_data['deformed'][point_ids]

        A_init=mesh_init.cell_data['area0'][i]

        A_deformed=getA(deformed_triangle_vertices)
        
        dA1,dA2,dA3=getdA(deformed_triangle_vertices,A_deformed)

        #prefactor=2*A_init*(A_init-A_deformed)
        prefactor=2*A_init*(A_init-A_deformed)/A_deformed/A_deformed+2*(A_init-A_deformed)*(A_init-A_deformed)/A_deformed/A_deformed/A_deformed
        derivatives[point_ids[0]]+=prefactor*dA1
        derivatives[point_ids[1]]+=prefactor*dA2
        derivatives[point_ids[2]]+=prefactor*dA3

    return derivatives

def calculateAreas(mesh_init:pv.PolyData):
    n_faces = mesh_init.n_faces
    Areas=np.zeros(n_faces)
    # Loop through all triangles
    for i,cell in enumerate(mesh_init.cell):
        triangle_vertices = cell.points
        x1=triangle_vertices[0,0]
        y1=triangle_vertices[0,1]
        z1=triangle_vertices[0,2]
        x2=triangle_vertices[1,0]
        y2=triangle_vertices[1,1]
        z2=triangle_vertices[1,2]
        x3=triangle_vertices[2,0]
        y3=triangle_vertices[2,1]
        z3=triangle_vertices[2,2]

        dx1=x2-x1
        dy1=y2-y1
        dz1=z2-z1

        dx2=x3-x1
        dy2=y3-y1
        dz2=z3-z1

        area=0.5*np.sqrt((dy1*dz2-dz1*dy2)*(dy1*dz2-dz1*dy2)
                            +(dz1*dx2-dx1*dz2)*(dz1*dx2-dx1*dz2)
                            +(dx1*dy2-dy1*dx2)*(dx1*dy2-dy1*dx2))
        Areas[i]=area
    mesh_init.cell_data['area0']=Areas

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

def findDiffeo(mesh_init,mesh_target,sub_manifolds_init,sub_manifolds_target):
    N_iterations=[30,20,10,10]
    start_rates=[1,0.2,0.1,0.05]
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
    
    for level in range(len(mesh_hirarchical)):
        calculateAreas(mesh_hirarchical[level])
        all_sum_deriv=[]
        #interative solving
        rate=start_rates[level]
        for i in range(N_iterations[level]):
            #print(i)
            #calculate energy and partial derivatives
            derivatives=getPartialDerivatives(mesh_hirarchical[level])

            #calculate normal component or take paralel component
            tangent_derivatives=getTangentComponent(derivatives,mesh_hirarchical[level],mesh_target,sub_manifolds_hirarchical[level])

            #do euler step
            max_deriv=np.max(tangent_derivatives)
            #print(max_deriv)
            sum_deriv=np.sum(np.linalg.norm(tangent_derivatives,axis=1))
            #print(sum_deriv)
            all_sum_deriv.append(sum_deriv)
            if i>2:
                if all_sum_deriv[-2]<all_sum_deriv[-1]:
                    print(rate)
                    rate=rate/1.5

            mesh_hirarchical[level].point_data["deformed"]=mesh_hirarchical[level].point_data["deformed"]+rate*tangent_derivatives/max_deriv
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
            return
        # p = pv.Plotter()
        # p.add_mesh(mesh_target, color='green', show_edges=True)
        # points_array=mesh_target.points[sub_manifolds_target[0].path]
        # polydata = pv.PolyData(points_array)
        # lines=np.zeros((len(points_array)+1),dtype=int)
        # lines[0]=len(points_array)
        # lines[1:] = np.arange(len(points_array),dtype=int)
        # polydata.lines = lines
        # p.add_mesh(polydata, color='green', line_width=5, render_lines_as_tubes=True)  
        # delta=30
        # mesh_hirarchical[level].points=mesh_hirarchical[level].point_data['deformed']-np.array((delta,0,0))
        # p.add_mesh(mesh_hirarchical[level], color='blue', show_edges=True)
        # points_array=mesh_hirarchical[level].point_data['deformed'][sub_manifolds_hirarchical[level][0].path]-np.array((delta,0,0))
        # polydata = pv.PolyData(points_array)
        # lines=np.zeros((len(points_array)+1),dtype=int)
        # lines[0]=len(points_array)
        # lines[1:] = np.arange(len(points_array),dtype=int)
        # polydata.lines = lines
        # p.add_mesh(polydata, color='blue', line_width=5, render_lines_as_tubes=True)  
        # delta=60
        # mesh_hirarchical[level+1].points=mesh_hirarchical[level+1].point_data['deformed']-np.array((delta,0,0))
        # p.add_mesh(mesh_hirarchical[level+1], color='blue', show_edges=True)
        # points_array=mesh_hirarchical[level+1].point_data['deformed'][sub_manifolds_hirarchical[level+1][0].path]-np.array((delta,0,0))
        # polydata = pv.PolyData(points_array)
        # lines=np.zeros((len(points_array)+1),dtype=int)
        # lines[0]=len(points_array)
        # lines[1:] = np.arange(len(points_array),dtype=int)
        # polydata.lines = lines
        # p.add_mesh(polydata, color='blue', line_width=5, render_lines_as_tubes=True)  
        # p.show()
    

    return