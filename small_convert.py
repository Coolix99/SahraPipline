import pandas as pd
import os

# Define the input file path and output file path
input_path = r"\\vs-grp07.zih.tu-dresden.de\max_kotz\sahra_shivani_data\sorted_data\for_curv_thick\scalarGrowthData.h5"
output_path = os.path.expanduser("~/Downloads/scalarGrowthData.csv")

# Read the HDF5 file and assume the data is in the default key
try:
    # Replace 'data_key' with the actual key if known
    data_key = None  
    with pd.HDFStore(input_path, mode='r') as hdf:
        if len(hdf.keys()) > 0:
            data_key = hdf.keys()[0]  # Use the first key as default

    if data_key is None:
        raise KeyError("No data found in the HDF5 file.")

    # Load the dataset
    df = pd.read_hdf(input_path, key=data_key)

    # Save it to a CSV file
    df.to_csv(output_path, index=False)
    print(f"File successfully converted and saved at: {output_path}")

except Exception as e:
    print(f"Error processing the file: {e}")
