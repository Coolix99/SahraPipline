import tifffile
import numpy as np

#from sklearn.cluster import DBSCAN
import os
import json
#import git

from config import *


def save_array_as_tiff(image_data: np.ndarray, output_tiff_file: str):
    # Save the NumPy array as a TIFF file
    tifffile.imwrite(output_tiff_file, image_data)

def getImage(file):
    with tifffile.TiffFile(file) as tif:
            try:
                image=tif.asarray()
            except:
                return None
            
            return image

def getImage_Meta(file):
    with tifffile.TiffFile(file) as tif:
        try:
            # Retrieve the image data
            image = tif.asarray()
            # Retrieve metadata
            if not tif.series[0].axes=='ZYX':
                print(tif.series[0].axes)
                print('unusual axis')
                return
          
            metadata = tif.pages[0].tags  # Access tags from the first page; adjust as necessary
            metadata_dict = {tag.name: tag.value for tag in metadata.values()}
   
            x_scale=metadata_dict['XResolution'][1]/metadata_dict['XResolution'][0]
            y_scale=metadata_dict['YResolution'][1]/metadata_dict['YResolution'][0]
            z_scale=tif.imagej_metadata['images']/tif.imagej_metadata['slices']
            if abs(z_scale-1)>0.001:
                print('z scale unusual')
                return
            return image, np.array((z_scale,y_scale,x_scale))
        except Exception as e:
            print(f"Failed to read TIFF file: {e}")
            return None, None

def saveArr(arr,path):
    np.save(path, arr)

def loadArr(path):
    return np.load(path+".npy")

def get_JSON(dir, name=None):
    if name is None:
        name = 'MetaData.json'
    else:
        name = f'MetaData_{name}.json'

    if not os.path.isdir(dir):  # Ensure dir is actually a valid directory
        print(f"Directory doesn't exist: {dir}")
        return {}

    json_file_path = os.path.join(dir, name)

    if not os.path.exists(json_file_path):  # Check if JSON file exists before opening
        print(f"MetaData doesn't exist: {dir}, {name}")
        return {}

    try:
        with open(json_file_path, 'r', encoding="utf-8") as json_file:
            return json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        print(f"[ERROR] Cannot read JSON: {json_file_path} -> {e}")
        return {}


def writeJSON(directory,key,value,name=None):
    if name is None:
        name='MetaData.json'
    else:
        name = 'MetaData_'+name+'.json'
    json_file_path=os.path.join(directory, name)
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}  # Create an empty dictionary if the file doesn't exist

    # Edit the values
    data[key] = value
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    # Close the file
    json_file.close()

def writeJSONlist(directory,keys,values,name=None):
    if name is None:
        name='MetaData.json'
    else:
        name = 'MetaData_'+name+'.json'
    json_file_path=os.path.join(directory, name)
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}  # Create an empty dictionary if the file doesn't exist

    # Edit the values
    for i,key in enumerate(keys):
        data[key] = values[i]
    
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    # Close the file
    json_file.close()

def make_path(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)

def exists_path(path):
    return os.path.exists(path)