import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import *
from IO import *


def main():
    # Initialize an empty list to collect all dataframes
    all_dfs = []

    # Iterate through folders
    EDprops_folder_list = [item for item in os.listdir(ED_cell_props_path) if os.path.isdir(os.path.join(ED_cell_props_path, item))]
    for EDprop_folder in EDprops_folder_list:
        print(EDprop_folder)
        EDprop_folder_path = os.path.join(ED_cell_props_path, EDprop_folder)

        EDpropMetaData = get_JSON(EDprop_folder_path)
        if not EDpropMetaData:
            print('No EDprops found')
            continue

        # Load the dataframe
        df_prop = pd.read_hdf(os.path.join(EDprop_folder_path, EDpropMetaData['MetaData_EDcell_props']['EDcells file']), key='data')

        # Add metadata as new columns
        df_prop['time in hpf'] = EDpropMetaData['MetaData_EDcell_props']['time in hpf']
        df_prop['condition'] = EDpropMetaData['MetaData_EDcell_props']['condition']

        # Append the dataframe to the list
        all_dfs.append(df_prop)

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Plot the violin plot for "Volume" split by time and condition
    plt.figure(figsize=(14, 8))

    sns.violinplot(
        data=merged_df,
        x='time in hpf',
        y='Volume',
        hue='condition',
        split=True,
        scale="width",
        inner=None,
        palette="Set2",
    )

    grouped = merged_df.groupby(['time in hpf', 'condition'])
    summary_stats = grouped['Volume'].agg(
        mean='mean',
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()

    # Overlay bars for mean, median, 25th, and 75th percentiles
    for _, row in summary_stats.iterrows():
        time = row['time in hpf']
        condition = row['condition']
        color = "orange" if condition == "Development" else "blue"  # Match violin plot colors

        # Mean
        plt.plot([time - 0.15, time + 0.15], [row['mean'], row['mean']], color=color, linestyle='-', linewidth=2)

        # Median
        plt.plot([time - 0.15, time + 0.15], [row['median'], row['median']], color=color, linestyle='--', linewidth=2)

        # 25th and 75th percentiles
        plt.plot([time - 0.1, time + 0.1], [row['q25'], row['q25']], color=color, linestyle=':', linewidth=1.5)
        plt.plot([time - 0.1, time + 0.1], [row['q75'], row['q75']], color=color, linestyle=':', linewidth=1.5)

    # Finalize the plot
    plt.title("Volume Distribution Over Time Split by Condition")
    plt.xlabel("Time in hpf")
    plt.ylabel("Volume")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Condition", labels=["Development", "Regeneration"], loc="upper right")
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
