import pyvista as pv
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
pd.set_option('display.max_colwidth', None)
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import find 

from config import *
from IO import *

def calculate_mesh_features(mesh:pv.PolyData):
    features = {}
    #print(mesh.point_data)
    features['surface_area'] = mesh.area
    features['average_curvature'] = mesh.point_data['avg_curvature'].mean()
    features['gauss_curvature'] = mesh.point_data['gauss_curvature'].mean()
    features['max average_curvature'] = np.abs(mesh.point_data['avg_curvature']).max()
    features['max gauss_curvature'] = np.abs(mesh.point_data['gauss_curvature']).max()
    features['L1'] = np.max(mesh.point_data['coord_1'])-np.min(mesh.point_data['coord_1'])
    features['L2'] = np.max(mesh.point_data['coord_2'])-np.min(mesh.point_data['coord_2'])
    features['aspect ratio'] = features['L1']/features['L2']


    return features

def getMST(df):
    if len(df)==1:
        return None

    meshes=[]
    for row in df.itertuples():
        meshes.append(pv.read(os.path.join(Surface_path,row.folder_name,row.file_name)))

    #calculate features
    features_list = [calculate_mesh_features(mesh) for mesh in meshes]
    feature_df = pd.DataFrame(features_list)
    scaler = StandardScaler()
    feature_df_normalized = scaler.fit_transform(feature_df)

    #PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(feature_df_normalized)
    pca_df = pd.DataFrame(data = principal_components, columns = ['principal_component_1', 'principal_component_2'])

    #Centroid
    centroid = pca_df.mean()
    pca_df['distance_from_centroid'] = np.sqrt((pca_df['principal_component_1'] - centroid['principal_component_1'])**2 + 
                                            (pca_df['principal_component_2'] - centroid['principal_component_2'])**2)
    pca_df['file_name'] = df['file_name'].values
    pca_df['folder_name'] = df['folder_name'].values
    pca_df = pca_df.reset_index(drop=True)
    sorted_df = pca_df.sort_values(by='distance_from_centroid')
    #print(sorted_df[['file_name', 'distance_from_centroid']])

    #MST
    coordinates = pca_df[['principal_component_1', 'principal_component_2']].values
    distance_matrix = squareform(pdist(coordinates, metric='euclidean'))
    mst_matrix = minimum_spanning_tree(distance_matrix)
    # Extracting edges with correct indexing
    rows, cols, distances = find(mst_matrix)

    edges_data = []
    for start, end, dist in zip(rows, cols, distances):
        if start != end:  # Ensure no self-loops, adjust logic as necessary
            edges_data.append({
                'Folder1': df.iloc[start]['folder_name'],
                'Folder2': df.iloc[end]['folder_name'],
                'Distance': mst_matrix[start, end]
            })
    # Convert list of dictionaries to DataFrame
    edges_df = pd.DataFrame(edges_data)
    #print(edges_df)

    # mst = mst_matrix.toarray().astype(float)
    # edges = np.column_stack(np.where(mst > 0))
    # # Plotting the points
    # plt.scatter(pca_df['principal_component_1'], pca_df['principal_component_2'], c='blue')
    # # Drawing the MST edges
    # for edge in edges:
    #     point1 = pca_df.iloc[edge[0]][['principal_component_1', 'principal_component_2']]
    #     point2 = pca_df.iloc[edge[1]][['principal_component_1', 'principal_component_2']]
    #     plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k-', alpha=0.6)

    # plt.title('Minimum Spanning Tree of Meshes in PCA Space')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.show()
    return edges_df

def make_sim_MSTs(relevant_conditions):
    relevant_conditions.sort()
    surface_folder_list=[folder for folder in os.listdir(Surface_path) if os.path.isdir(os.path.join(Surface_path, folder))]
    data_list = []
    for surface_folder in surface_folder_list:
        surface_dir_path=os.path.join(Surface_path,surface_folder)
        MetaData=get_JSON(surface_dir_path)
        if not 'Coord_MetaData' in MetaData:
            continue
        MetaData=MetaData['Coord_MetaData']
        print(surface_folder)
        data_list.append({
            'folder_name':surface_folder,
            'file_name': MetaData['Surface file'],
            'hpf': MetaData['time in hpf'],
            'control': MetaData['is control'],
            'genotype': MetaData['genotype'],
            'experimentalist': MetaData['experimentalist']
        })
    df = pd.DataFrame(data_list)
    grouped_df = df.groupby(relevant_conditions)

    combined_string = '_'.join(relevant_conditions)
    MST_path_dir=os.path.join(SimilarityMST_path,combined_string)
    make_path(MST_path_dir)

    data_list = []
    for name, group in grouped_df:
        print("Group Name:", name)
        print(group, "\n")
        edges_df = getMST(group)
        if edges_df is None:
            continue
        combined_string = '_'.join(str(item) for item in name)
        edges_df.to_hdf(os.path.join(MST_path_dir,combined_string+'.h5'), key='data', mode='w')
        data_list.append({c:n for c,n in zip(relevant_conditions,name)})
        data_list[-1]['file name']=combined_string+'.h5'
    df = pd.DataFrame(data_list)
    print(df)
    df.to_hdf(os.path.join(MST_path_dir,'alldf.h5'), key='data', mode='w')
    return
    print(df)
    df = df[df['experimentalist'] == 'Heewon']
    df = df[df['hpf'] == 72]
    
    

def main():
    make_sim_MSTs(['experimentalist', 'hpf','control','genotype'])



    return


    plt.figure(figsize=(8,6))
    scatter = plt.scatter(pca_df['principal_component_1'], pca_df['principal_component_2'], alpha=0.5)
    for i, row in enumerate(df.itertuples()):
        plt.annotate(row.file_name, (pca_df['principal_component_1'][i], pca_df['principal_component_2'][i]))
    plt.title('2D Visualization of Meshes Using PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    return
    print(feature_df)
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=200)
    tsne_results = tsne.fit_transform(feature_df)

    # Extract the 2D coordinates
    feature_df['tsne-2d-one'] = tsne_results[:,0]
    feature_df['tsne-2d-two'] = tsne_results[:,1]


    plt.figure(figsize=(8,6))
    scatter = plt.scatter(feature_df['tsne-2d-one'], feature_df['tsne-2d-two'], alpha=0.5)

    for i, row in enumerate(df.itertuples()):
        plt.annotate(row.file_name, (feature_df['tsne-2d-one'][i], feature_df['tsne-2d-two'][i]))

    plt.title('2D Visualization of Meshes')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.show()

if __name__ == "__main__":

    main()