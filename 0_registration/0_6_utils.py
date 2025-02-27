import os
import json
import shutil

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *


def contains_string(data, target_string):
    """Recursively check if a dictionary contains the target string."""
    if isinstance(data, dict):
        return any(contains_string(v, target_string) for v in data.values())
    elif isinstance(data, list):
        return any(contains_string(v, target_string) for v in data)
    elif isinstance(data, str):
        return target_string in data
    return False

def find_and_delete_folders(root_folder, target_string, dry_run=True):
    """Recursively find 'MetaData.json' files and delete the containing folder if string is found."""
    for dirpath, _, filenames in os.walk(root_folder, topdown=False):  # bottom-up to delete safely
        if "MetaData.json" in filenames:
            metadata_path = os.path.join(dirpath, "MetaData.json")

            try:
                with open(metadata_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                if contains_string(data, target_string):
                    if dry_run:
                        print(f"[DRY RUN] Would delete: {dirpath}")
                    else:
                        print(f"Deleting: {dirpath}")
                        shutil.rmtree(dirpath)  # Remove the folder and its contents

            except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
                print(f"Error processing {metadata_path}: {e}")

# Example usage
root_directory = Output_path
search_string = "4850cut"  # Change this to the string you're looking for
dry_run_mode = False  # Set to False to actually delete files

find_and_delete_folders(root_directory, search_string, dry_run_mode)
