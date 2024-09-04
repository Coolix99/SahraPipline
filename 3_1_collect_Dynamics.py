from functools import reduce

from config import *
from IO import *

def main():
    finmasks_folder_list= [item for item in os.listdir(finmasks_path) if os.path.isdir(os.path.join(finmasks_path, item))]
    for mask_folder in finmasks_folder_list:
        print(mask_folder)
        mask_folder_path=os.path.join(finmasks_path,mask_folder)
        
        maskMetaData=get_JSON(mask_folder_path)
        if maskMetaData=={}:
            print('no mask')
            continue
        membrane_folder_path=os.path.join(membranes_path,mask_folder)
        MetaData=get_JSON(membrane_folder_path)
        mask_img=getImage(os.path.join(mask_folder_path,maskMetaData['MetaData_finmasks']['finmasks file']))
        if  MetaData=={}:
            MetaData['MetaData_membrane']=maskMetaData['MetaData_finmasks']
        MetaData=MetaData['MetaData_membrane']
        voxel_size=reduce(lambda x, y: x * y, MetaData['scales ZYX'])
        Volume=np.sum(mask_img>0)*voxel_size
        print(Volume)
        # MetaData['scales ZYX']
        # MetaData['condition']
        # MetaData['time in hpf']
        # MetaData['experimentalist']
        # MetaData['genotype']

        #TODO do stuff with surface
        #TODO fill df and safe is
        

    #TODO V(t),A(t),Axis(t)?,V(t) by integrating d, d_mean=V/A

if __name__ == "__main__":
    main()