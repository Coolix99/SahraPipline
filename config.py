import os

from config_machine import *

if(machine=='Home'):
    #Sahra_Shivani_path=(r'E:\02_Data\share_Sahra{}').format("")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    

if(machine=='Laptop'):
    #Sahra_Shivani_path=(r'C:\Users\s0095413\Documents\02_Data\share_Sarah\{}').format("")
    script_dir = os.path.dirname(os.path.abspath(__file__))

if(machine=='BA'):
    Sahra_Shivani_path=(r'/home/max/Documents/02_Data/sahra_shivani_data')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    

"""path structure"""
gitPath=script_dir

Output_path=os.path.join(Sahra_Shivani_path,'sorted_data')

#Part 0
Input_Shivani_path=os.path.join(Sahra_Shivani_path,'from_Shivani')
membranes_path=os.path.join(Output_path,'membranes')
ED_marker_path=os.path.join(Output_path,'ED_marker')
finmasks_path=os.path.join(Output_path,'finmasks')

# #Part I
FlatFin_path=os.path.join(Output_path,'for_curv_thick','FlatFin')

#Part II
primitive_Graph_path=os.path.join(Output_path,'for_curv_thick','Diffeo','primitive_Graph')
# ElementaryDiffeos_path=os.path.join(Output_path,'for_curv_thick','Diffeo','ElementaryDiffeos')
# Hist_path=os.path.join(Output_path,'for_curv_thick','Diffeo','Histogramms')

# #Part III
# EpFlat_path=os.path.join(Input_path,'epithelia_flat_test')
# EpSeg_path=os.path.join(Output_path,'epithelia_flat_test')
# EpSeg_path=os.path.join(EpSeg_path,'fin')

"""versions"""
#Part I
Orient_version = 1
CenterLine_version = 1
Surface_version = 1
Thickness_version = 2
Coord_version = 1

#Part II
Diffeo_version = 1
Hist_version = 1
Mesh_version = 1

#Part III
project_version = 1
apply_version = 1
seg_version = 1
morpho_version=1