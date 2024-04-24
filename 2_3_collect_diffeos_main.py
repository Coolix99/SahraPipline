import pandas as pd

from config import *
from IO import *


def main():
    ED_folder_list=[folder for folder in os.listdir(ElementaryDiffeos_path) if os.path.isdir(os.path.join(ElementaryDiffeos_path, folder))]
    data_list=[]
    for grouping in ED_folder_list:
        grouping_path=os.path.join(ElementaryDiffeos_path,grouping)
        group_folder_list=[folder for folder in os.listdir(grouping_path) if os.path.isdir(os.path.join(grouping_path, folder))]
        for group in group_folder_list:
            group_path=os.path.join(grouping_path,group)
            diffeo_list=[folder for folder in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, folder))]
            for diff_folder in diffeo_list:
                diff_path=os.path.join(group_path,diff_folder)
                MetaData=get_JSON(diff_path)
                if not 'MetaData_Diffeo' in MetaData:
                    continue
                data_list.append({
                    'grouping':grouping,
                    'group':group,
                    'diff_folder':diff_folder,
                    'init_folder':MetaData['MetaData_Diffeo']['init_folder'],
                    'target_folder':MetaData['MetaData_Diffeo']['target_folder']
                })
    df = pd.DataFrame(data_list)
    df.to_hdf(os.path.join(ElementaryDiffeos_path,'alldf.h5'), key='data', mode='w')

if __name__ == "__main__":
    #test()
    main()