import os

from config_machine import *

if(machine=='Home'):
    structured_data_path=(r'\\vs-grp07.zih.tu-dresden.de\max_kotz\structured_data\{}').format("")
    structured_data_path=(r'E:\02_Data\structured_data\{}').format("")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
#     gitPath=(r'C:\Users\Admin\Documents\code\cell_properties_extraction')

if(machine=='Laptop'):
    structured_data_path=(r'C:\Users\s0095413\Documents\02_Data\structured_data\{}').format("")
    script_dir = os.path.dirname(os.path.abspath(__file__))

if(machine=='BA'):
    structured_data_path=(r'/home/max/Documents/02_Data/structured_data/{}').format("")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    

"""path structure"""
gitPath=script_dir

nuclei_images_path=os.path.join(structured_data_path,'images','RAW_images_and_splitted','raw_images_nuclei')
vol_images_path=os.path.join(structured_data_path,'images','vol_and_nuclei_mask','raw_images_vol')

LMcoord_path=os.path.join(structured_data_path,'images','fin_geometry','LM_coord')

"""versions"""
Orient_version = 2
CenterLine_version = 2
Surface_version = 2
Thickness_version = 1
Coord_version = 1
