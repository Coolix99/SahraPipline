import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

def getData():
    df_file_path = os.path.join(scalar_path,'scalarGrowthData_meshBased.csv')

    # Load the DataFrame from the HDF5 file
    df = pd.read_csv(df_file_path,sep=',')

   
    
    # select only the columns we need
    df = df[['time in hpf', 'condition', 'Surface Area']]

    

    return df

