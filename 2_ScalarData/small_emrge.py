import pandas as pd
from pathlib import Path

new_csv = Path("/home/max/Downloads/membrane_dynamics_removesmallfins_final.csv")
old_csv = Path("/media/grp07_max/sahra_shivani_data/sorted_data/scalarData/scalarGrowthData_meshBased.csv")
out_csv = Path("/media/grp07_max/sahra_shivani_data/sorted_data/scalarData/membrane_Data.csv")

key_col = "Mask Folder"

old_df = pd.read_csv(old_csv)
new_df = pd.read_csv(new_csv)

# Use the old dataframe schema/order.
# Extra columns in the new dataframe, e.g. Mean_thickness, are ignored.
old_cols = old_df.columns.tolist()

# Add missing old columns to new data, including ED columns, as empty values.
for col in old_cols:
    if col not in new_df.columns:
        new_df[col] = pd.NA

new_df = new_df[old_cols]

# Remove old rows whose Mask Folder is present in new data,
# then append the new rows so they overwrite old entries.
new_keys = set(new_df[key_col].astype(str))
old_keep = old_df[~old_df[key_col].astype(str).isin(new_keys)]

merged_df = pd.concat([old_keep, new_df], ignore_index=True)

# Optional: sort by Mask Folder for reproducible output
merged_df = merged_df.sort_values(key_col).reset_index(drop=True)

merged_df.to_csv(out_csv, index=False)

print(f"Saved merged data to: {out_csv}")
print(f"Old rows: {len(old_df)}")
print(f"New rows: {len(new_df)}")
print(f"Merged rows: {len(merged_df)}")
print(f"Overwritten rows: {len(old_df) - len(old_keep)}")