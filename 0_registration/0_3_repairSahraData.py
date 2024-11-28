import os
import json
import re
from typing import Tuple

def extract_info(s: str) -> Tuple[int, bool]:
    # Extract the number before 'h'
    match = re.search(r'(\d+)h', s)
    if match:
        hours = int(match.group(1))
    else:
        raise ValueError("The string does not contain a valid 'h' pattern with a preceding number.")
    
    # Check if '_reg' is present in the string
    if 'reg' in s and 'dev' in s:
        raise ValueError("Both 'reg' and 'dev' are present in the string.")
    elif 'reg' in s:
        return hours, False
    elif 'dev' in s:
        return hours, True
    else:
        raise ValueError("Neither 'reg' nor 'dev' are present in the string.")

def repairMetaData(file_path):
    """
    Function to repair the MetaData.json file.
    
    Parameters:
    -----------
    file_path : str
        The path to the MetaData.json file that needs to be repaired.
    
    Returns:
    --------
    None
    """
    with open(file_path, 'r') as file:
        metadata = json.load(file)
    
    modified = False  # Flag to track if any changes are made
    
    # Iterate through the metadata keys
    for key, value in metadata.items():
        # Check if the required fields exist
        if 'experimentalist' in value and value['experimentalist'] == 'Sahra':
            if 'condition' in value:
                # Look for fields containing 'file' and check the condition
                for k, v in value.items():
                    if 'file' in k:
                        try:
                            _, is_dev = extract_info(v)
                            if is_dev and value['condition'] != 'Development':
                                print(f"Updating 'condition' to 'Development' for {v}")
                                value['condition'] = 'Development'
                                modified = True
                            elif not is_dev and value['condition'] != 'Regeneration':
                                print(f"Updating 'condition' to 'Regeneration' for {v}")
                                value['condition'] = 'Regeneration'
                                modified = True
                        except ValueError as e:
                            print(f"Error processing file: {v}, {e}")

    # If the metadata was modified, save it back to the same file
    if modified:
        with open(file_path, 'w') as file:
            json.dump(metadata, file, indent=4)
        print(f"MetaData file {file_path} updated and saved.")

# Function to walk through directories and apply repairMetaData on MetaData.json files
def walk_and_repair(start_folder):
    """
    Walk through all subfolders starting from 'start_folder', and apply the 'repairMetaData'
    function to any 'MetaData.json' files found.
    
    Parameters:
    -----------
    start_folder : str
        The root folder to start walking through subfolders.
    
    Returns:
    --------
    None
    """
    for root, dirs, files in os.walk(start_folder):
        # Check if 'MetaData.json' is present in the current directory
        if 'MetaData.json' in files:
            metadata_file_path = os.path.join(root, 'MetaData.json')
            print(f"Found MetaData.json: {metadata_file_path}")
            
            # Apply the repairMetaData function to this file
            repairMetaData(metadata_file_path)

# Example usage
if __name__ == "__main__":
    from config import Output_path
    # Call the walk_and_repair function
    walk_and_repair(Output_path)
