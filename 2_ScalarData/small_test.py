
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from IO import *
csv_file = os.path.join(scalar_path, 'scalarGrowthData_meshBased.csv')

df = pd.read_csv(csv_file)
print(df.columns)
print(df['genotype'].unique())
print(df['condition'].unique())

# import matplotlib.pyplot as plt

# # filter condition
# df_sel = df[df['condition'] == 'Smoc12_Dev']

# # --- Volume vs time ---
# plt.figure()
# plt.scatter(df_sel['time in hpf'], df_sel['Volume'])
# plt.xlabel('time in hpf')
# plt.ylabel('Volume')
# plt.title('Smoc12_Dev: Volume vs Time')
# plt.show()

# # --- Surface Area vs time ---
# plt.figure()
# plt.scatter(df_sel['time in hpf'], df_sel['Surface Area'])
# plt.xlabel('time in hpf')
# plt.ylabel('Surface Area')
# plt.title('Smoc12_Dev: Surface Area vs Time')
# plt.show()

df_smoc_dev = pd.read_excel(
        os.path.join(scalar_path, "Smoc1_Smoc2_dev.xlsx")
    )

print(df_smoc_dev.columns)
print(df_smoc_dev['Genotype'].unique())
print(df_smoc_dev['Is control'].unique())
print(df_smoc_dev['Experimentalist'].unique())

import matplotlib.pyplot as plt

# --- filter first dataframe ---
df_sel = df[df['condition'] == 'Smoc12_Dev']

# --- filter second dataframe ---
df_vinita = df_smoc_dev[df_smoc_dev['Experimentalist'] == 'Vinita']
df_lucas  = df_smoc_dev[df_smoc_dev['Experimentalist'] == 'Lucas']

# --- Volume vs time ---
plt.figure()

# dataset 1
plt.scatter(
    df_sel['time in hpf'],
    df_sel['Volume'],
    label='meshBased (Smoc12_Dev)',
)

# dataset 2 - Vinita
plt.scatter(
    df_vinita['Time in hpf'],
    df_vinita['Volume'],
    label='excel (Vinita)',
)

# dataset 3 - Lucas
plt.scatter(
    df_lucas['Time in hpf'],
    df_lucas['Volume'],
    label='excel (Lucas)',
)

plt.xlabel('time in hpf')
plt.ylabel('Volume')
plt.title('Volume vs Time')
plt.legend()
plt.show()


# --- Surface Area vs time ---
plt.figure()

plt.scatter(
    df_sel['time in hpf'],
    df_sel['Surface Area'],
    label='meshBased (Smoc12_Dev)',
)

plt.scatter(
    df_vinita['Time in hpf'],
    df_vinita['surface_area'],
    label='excel (Vinita)',
)

plt.scatter(
    df_lucas['Time in hpf'],
    df_lucas['surface_area'],
    label='excel (Lucas)',
)

plt.xlabel('time in hpf')
plt.ylabel('Surface Area')
plt.title('Surface Area vs Time')
plt.legend()
plt.show()

# --- prepare first dataframe (meshBased) ---
df1 = df_sel.copy()

df1_out = pd.DataFrame({
    'Name': df1['Mask Folder'],
    'Volume': df1['Volume'],
    'Surface Area': df1['Surface Area'],
    'Genotype': df1['genotype'],
    'Experimentalist': df1['experimentalist'],
    'hpf': df1['time in hpf'],
})

df1_out['description'] = 'smoc12_dev_vinita_new'


# --- prepare Vinita (excel) ---
df2 = df_vinita.copy()

df2_out = pd.DataFrame({
    'Name': df2['Name'],
    'Volume': df2['Volume'],
    'Surface Area': df2['surface_area'],
    'Genotype': df2['Genotype'],
    'Experimentalist': df2['Experimentalist'],
    'hpf': df2['Time in hpf'],
})

df2_out['description'] = 'smoc12_dev_vinita_old'


# --- prepare Lucas (excel) ---
df3 = df_lucas.copy()

df3_out = pd.DataFrame({
    'Name': df3['Name'],
    'Volume': df3['Volume'],
    'Surface Area': df3['surface_area'],
    'Genotype': df3['Genotype'],
    'Experimentalist': df3['Experimentalist'],
    'hpf': df3['Time in hpf'],
})

df3_out['description'] = 'smoc12_dev_lucas'


# --- merge all ---
df_merged = pd.concat([df1_out, df2_out, df3_out], ignore_index=True)


# --- save ---
# out_file = os.path.join(scalar_path, 'merged_smoc12_dev.csv')
df_merged.to_csv('./merged_smoc12_dev.csv', index=False)

# print("Saved to:", out_file)
print(df_merged.head())