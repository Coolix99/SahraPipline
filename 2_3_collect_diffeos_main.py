import pandas as pd
import pyvista as pv

from config import *
from IO import *


def main():
    ED_folder_list=[folder for folder in os.listdir(ElementaryDiffeos_path) if os.path.isdir(os.path.join(ElementaryDiffeos_path, folder))]
    data_list=[]
    for ED_folder in ED_folder_list:
        ED_folder_path=os.path.join(ElementaryDiffeos_path,ED_folder)
        MetaData=get_JSON(ED_folder_path)
        if not 'MetaData_Diffeo' in MetaData:
            continue
        data_list.append({
            'diff_folder':ED_folder,
            'init_folder':MetaData['MetaData_Diffeo']['init_folder'],
            'target_folder':MetaData['MetaData_Diffeo']['target_folder']
        })
    df = pd.DataFrame(data_list)
    df.to_hdf(os.path.join(ElementaryDiffeos_path,'alldf.h5'), key='data', mode='w')

def plot():
    ED_folder_list=[folder for folder in os.listdir(ElementaryDiffeos_path) if os.path.isdir(os.path.join(ElementaryDiffeos_path, folder))]
    #ED_folder_list=['diffeo_0a0502224f']
    for ED_folder in ED_folder_list:
        print(ED_folder)
        ED_folder_path=os.path.join(ElementaryDiffeos_path,ED_folder)
        MetaData=get_JSON(ED_folder_path)
        if not 'MetaData_Diffeo' in MetaData:
            continue
        
        Diffeo=np.load(os.path.join(ED_folder_path,MetaData['MetaData_Diffeo']["Diffeo file"]))

        init_folder=MetaData['MetaData_Diffeo']['init_folder']
        target_folder=MetaData['MetaData_Diffeo']['target_folder']

        init_folder_dir=os.path.join(FlatFin_path,init_folder)
        init_mesh=pv.read(os.path.join(init_folder_dir,get_JSON(init_folder_dir)['Thickness_MetaData']['Surface file']))

        target_folder_dir=os.path.join(FlatFin_path,target_folder)
        target_mesh=pv.read(os.path.join(target_folder_dir,get_JSON(target_folder_dir)['Thickness_MetaData']['Surface file']))

        print(Diffeo.shape)
        print(init_mesh)
        print(target_mesh)
        # init_mesh.plot()
        # target_mesh.plot()
        init_mesh.points=Diffeo
        # init_mesh.plot()
        plotter = pv.Plotter()
        plotter.add_mesh(target_mesh, color='blue', label='Target')
        plotter.add_mesh(init_mesh, color='red', label='Defomed')
        plotter.add_legend()
        plotter.show()


if __name__ == "__main__":
    main()
    #plot()