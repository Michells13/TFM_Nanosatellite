import os
import cv2
import numpy as np
import shutil
from PIL import Image
import cv2
from utils import createLocations , write_subdataset, convertTif2Jpg

#Define paths
path_in = "/home/msiau/workspace/cloudsen12/subset/"
path_out= "/home/msiau/workspace/cloudsen12/split/"

#define type of label to use
type_of_label= "manual_hq"

#create folders if they don't exist 
createLocations(path_out)


#check =it out how manny subfolders there are
subfolders = [f for f in os.listdir(path_in) if os.path.isdir(os.path.join(path_in, f))]
i=0
#split the data into test and train  by a 0.X factor 
train=int(len(subfolders)*0.8)

#locate the data in the database structure of tensorflow
for subfolder in subfolders:
    i=i+1
    if i<train:
    
        write_subdataset(path_in + subfolder ,path_out+"/train",type_of_label)
    else:
        write_subdataset(path_in + subfolder ,path_out+"/test",type_of_label)
        
convertTif2Jpg(path_out)








