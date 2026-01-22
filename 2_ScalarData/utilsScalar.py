import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

def getData():
    df_file_path = os.path.join(scalar_path,'scalarGrowthData_meshBased.csv')

    # Load the DataFrame from the HDF5 file
    df = pd.read_csv(df_file_path,sep=';')

    # Calculate new columns:
    # L AP * L PD
    df['L AP * L PD'] = df['L AP'] * df['L PD']

    # V / A (Volume / Surface Area)
    df['V / A'] = df['Volume'] / df['Surface Area']

    # Int_dA_d / A (Integrated Thickness / Surface Area)
    df['Int_dA_d / A'] = df['Integrated Thickness'] / df['Surface Area']

    df['log L AP'] = np.log(df['L AP'])
    df['log L PD'] = np.log(df['L PD'])

    #with ED
    df['log L AP ED'] = np.log(df['L AP ED'])
    df['log L PD ED'] = np.log(df['L PD ED'])
    df['L AP ED * L PD ED'] = df['L AP ED'] * df['L PD ED']
    df['Thickness ED'] = df['ED Mask Volume']/df['Surface Area ED']
    df['Thickness ED+muscle+epithelium'] = df['Integrated Thickness ED']/df['Surface Area ED']
    df['Cell volume density ED'] = df['N ED cells']/df['ED Mask Volume']
    df['Cell area density ED'] = df['N ED cells']/df['Surface Area ED']

    df['A ED/A'] = df['Surface Area ED']/df['Surface Area']
    df['V ED/V'] = df['ED Mask Volume']/df['Volume']
    
    return df

