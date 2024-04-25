import pandas as pd

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

if __name__ == "__main__":
    main()