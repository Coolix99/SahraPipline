import pandas as pd

df = pd.read_csv("/media/grp07_max/sahra_shivani_data/sorted_data/scalarData/Lucas_Vinita_smoc_merged.csv")

print("\n=== Total ===")
print(df["condition"].value_counts())

print("\n=== By hpf ===")
counts = (
    df.groupby(["time in hpf", "condition"])
      .size()
      .unstack(fill_value=0)
      .sort_index()
)
print(counts)