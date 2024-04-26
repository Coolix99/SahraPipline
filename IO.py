import tifffile
import numpy as np

#from sklearn.cluster import DBSCAN
import os
import json
#import git

from config import *



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

def get_JSON(dir,name=None):
    if name is None:
        name='MetaData.json'
    else:
        name = 'MetaData_'+name+'.json'
    json_file_path=os.path.join(dir, name)
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print("MetaData doesn't exist", dir, name)
        data = {}  # Create an empty dictionary if the file doesn't exist
    return data

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