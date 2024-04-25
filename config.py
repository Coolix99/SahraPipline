import os

from config_machine import *

if(machine=='Home'):
    Sahra_path=(r'\\vs-grp07.zih.tu-dresden.de\max_kotz\structured_data\{}').format("")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    

if(machine=='Laptop'):
    Sahra_path=(r'\\vs-grp07.zih.tu-dresden.de\max_kotz\structured_data\{}').format("")
    script_dir = os.path.dirname(os.path.abspath(__file__))

if(machine=='BA'):
    Sahra_path=(r'/home/max/Documents/02_Data/share_Sarah/{}').format("")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    

"""path structure"""
gitPath=script_dir

Input_path=os.path.join(Sahra_path,'for_max')
Output_path=os.path.join(Sahra_path,'for_sahra')

#Part I
vol_path=os.path.join(Input_path,'for_curv_thick')
FlatFin_path=os.path.join(Output_path,'for_curv_thick','FlatFin')

#Part II
SimilarityMST_path=os.path.join(Output_path,'for_curv_thick','Diffeo','SimilarityMST')
ElementaryDiffeos_path=os.path.join(Output_path,'for_curv_thick','Diffeo','ElementaryDiffeos')
Hist_path=os.path.join(Output_path,'for_curv_thick','Diffeo','Histogramms')

"""versions"""
#Part I
Orient_version = 1
CenterLine_version = 1
Surface_version = 1
Thickness_version = 1
Coord_version = 1

#Part II
Diffeo_version = 1
Hist_version = 1