import os

from config_machine import *

if(machine=='Home'):
    Sahra_Shivani_path=(r'\\vs-grp07.zih.tu-dresden.de\max_kotz\sahra_shivani_data')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    

if(machine=='Laptop'):
    #Sahra_Shivani_path=(r'C:\Users\s0095413\Documents\02_Data\share_Sarah\{}').format("")
    script_dir = os.path.dirname(os.path.abspath(__file__))

if(machine=='BA'):
    #Sahra_Shivani_path=(r'/home/max/Documents/02_Data/sahra_shivani_data')
    Sahra_Shivani_path=(r'/media/max_kotz/sahra_shivani_data')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    

"""path structure"""
gitPath=script_dir

Output_path=os.path.join(Sahra_Shivani_path,'sorted_data')

#Part 0
Input_Shivani_path=os.path.join(Sahra_Shivani_path,'from_Shivani')
Input_Sahra_path=os.path.join(Sahra_Shivani_path,'from_Sahra')

membranes_path=os.path.join(Output_path,'membranes')
ED_marker_path=os.path.join(Output_path,'ED_marker')
ED_cells_path=os.path.join(Output_path,'ED_cells')
ED_cell_props_path=os.path.join(Output_path,'ED_cell_props')
finmasks_path=os.path.join(Output_path,'finmasks')

# #Part I
FlatFin_path=os.path.join(Output_path,'for_curv_thick','FlatFin')

#Part II
Curv_Thick_path=os.path.join(Output_path,'for_curv_thick')
Diffeo_path=os.path.join(Curv_Thick_path,'Diffeo')
ElementaryDiffeos_path=os.path.join(Diffeo_path,'ElementaryDiffeos')
Hist_path=os.path.join(Diffeo_path,'Histogramms')

# #Part III
Lucas_res=os.path.join(Output_path,'compare_Lucas')
AvgShape_path=os.path.join(Curv_Thick_path,'AvgShape')
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