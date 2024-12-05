import pandas as pd

from config import *
from IO import *

def getData():
    hdf5_file_path = os.path.join(Curv_Thick_path, 'scalarGrowthData.h5')

    # Load the DataFrame from the HDF5 file
    df = pd.read_hdf(hdf5_file_path, key='data')

    # # Calculate new columns:
    # # L AP * L PD
    # df['L AP * L PD'] = df['L AP'] * df['L PD']

    # # V / A (Volume / Surface Area)
    # df['V / A'] = df['Volume'] / df['Surface Area']

    # # Int_dA_d / A (Integrated Thickness / Surface Area)
    # df['Int_dA_d / A'] = df['Integrated Thickness'] / df['Surface Area']

    # df['log L AP'] = np.log(df['L AP'])
    # df['log L PD'] = np.log(df['L PD'])


    return df

def getCompareData():
    file_path=os.path.join('/media/max_kotz/joint_results/from_Lucas/Volume/Membrane_SS','fin_vol.csv')
    data = pd.read_csv(file_path)

    data = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
    data = data.rename(columns={'volume': 'volume_compare'})
    return data

def main():
    df=getData()
    #print(df)

    comp_df=getCompareData()
    #print(comp_df)
        
    condition_to_group = {'Development': 'dev', 'Regeneration': 'reg'}
    df['group'] = df['condition'].map(condition_to_group)

    # Perform the join based on the conditions
    merged_df = pd.merge(
        df,
        comp_df,
        left_on=['Mask Folder', 'time in hpf', 'group'],
        right_on=['filename', 'time', 'group'],
        how='inner'
    )

    # Drop the unnecessary 'group' column from the result
    merged_df = merged_df.drop(columns=['group'])

    merged_df['diff']=np.abs(merged_df['Volume']-merged_df['volume_compare'])/(merged_df['Volume']+merged_df['volume_compare'])

    sorted_df = merged_df.sort_values(by='diff', ascending=False)

    # Select the desired columns
    result = sorted_df[['Mask Folder', 'Volume', 'volume_compare', 'condition', 'time in hpf','diff']]

    pd.set_option('display.max_rows', None)  
    #pd.set_option('display.max_columns', None)
    print(result)

def test():
    impath1=os.path.join('/media/max_kotz/sahra_shivani_data/sorted_data/finmasks/160624_reg_sox10_claudin-gfp_72h_pecfin1','160624_reg_sox10_claudin-gfp_72h_pecfin1.tif')
    impath2=os.path.join('/media/max_kotz/sahra_shivani_data/from_Shivani/tissue_masks/masks_72hpf_reg','160624_reg_sox10_claudin-gfp_72h_pecfin1.tif')

    im1=getImage(impath1)
    im2=getImage(impath2)

    print(np.sum(im1>0)*0.7*0.2075664552409598*0.2075664552409598)
    print(np.sum(im2>0)*0.7*0.2075664552409598*0.2075664552409598)

if __name__ == "__main__":
    #main()

    test()