import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

def getData():
    df_file_path = os.path.join(scalar_path,'scalarGrowthData.csv')

    # Load the DataFrame from the HDF5 file
    df = pd.read_csv(df_file_path,sep=';')

    # remove lines where condition is '4850cut' or '7230cut'
    df = df[~df['condition'].isin(['4850cut', '7230cut'])]
    
    # select only the columns we need
    df = df[['time in hpf', 'condition', 'Surface Area']]

    #resclae area 
    df['Surface Area'] = df['Surface Area'] / 10000

    return df

