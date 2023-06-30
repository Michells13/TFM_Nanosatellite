import os
import cv2
import numpy as np
import shutil
from PIL import Image
import cv2

def convert_images(folder_path):
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tif') or file_name.endswith('.tiff'):
            # Generate the file paths
            tiff_path = os.path.join(folder_path, file_name)
            jpg_path = os.path.join(folder_path, file_name.replace('.tif', '.jpg').replace('.tiff', '.jpg'))

            # Read the TIFF image
            tiff_image = cv2.imread(tiff_path)

            # Write the image as JPG
            cv2.imwrite(jpg_path, tiff_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            # Delete the TIFF image
            os.remove(tiff_path)



def count_files_in_folder(folder_path):
    file_count = 0

    for _, _, files in os.walk(folder_path):
        file_count += len(files)

    return file_count
def copy_and_rename_image(source_path, destination_folder, new_filename):
    if os.path.exists(source_path):
        # Get the file extension from the source path
        _, extension = os.path.splitext(source_path)
        
        # Construct the new destination path with the renamed file
        destination_path = os.path.join(destination_folder, new_filename + extension)
        
        # Copy the image file to the destination folder with the new name
        shutil.copy2(source_path, destination_path)
        #convert_tif_to_png(source_path, destination_folder, new_filename)
        print("Image copied and renamed successfully.")
    else:
        print("Source image does not exist.")
def copy_image(source_path, destination_path):
    if os.path.exists(source_path):
        shutil.copy2(source_path, destination_path)
        print("Image copied successfully.")
    else:
        print("Source image does not exist.")
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Folder created:", path)
    else:
        print("Folder exists:", path)
def calculate_pixel_percentage(image):
    if image.ndim == 2:
        # Single-channel image (grayscale)
        gray_image = image
    elif image.ndim == 3:
        # Multi-channel image (BGR or BGRA)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid image dimensions.")

    # Calculate the total number of pixels
    total_pixels = gray_image.size

    # Count the number of white pixels
    white_pixels = cv2.countNonZero(gray_image)

    # Calculate the percentage of white and black pixels
    white_percentage = (white_pixels / total_pixels) * 100
    black_percentage = 100 - white_percentage

    return white_percentage, black_percentage

def eval_clouds(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        white_percentage, black_percentage = calculate_pixel_percentage(img)
        return white_percentage, black_percentage
    else:
        print("Failed to read the image:", path)
        return None, None
    print(white_percentage)

def write_subdataset(path,path_out,annotationType):
    subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    i=0
    for subfolder in subfolders:
        subfolder_path = os.path.join(path, subfolder)
        subsubfolders= [f for f in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, f))]

        for subsubfolder in subsubfolders:
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
            annotation_path= subsubfolder_path + "/labels/" + annotationType + ".tif"
            thumbnail_path=  subsubfolder_path + "/" + "thumbnail.tif"
            white_percentage, black_percentage =eval_clouds(annotation_path)
            i=i+1
            # print("Subfolder:", subsubfolder_path)
            # print("annotation path: ",annotation_path)
            # print ("thumbnail path: ",thumbnail_path )
            # print(white_percentage)
            if white_percentage < 25 : 
                copy_and_rename_image(thumbnail_path,path_out + "/clouds0_25/", str(count_files_in_folder(path_out + "/clouds0_25/"))) 
            if white_percentage < 50 and white_percentage >= 25: 
                copy_and_rename_image(thumbnail_path,path_out + "/clouds25_50/", str(count_files_in_folder(path_out + "/clouds25_50/"))) 
            if white_percentage < 75 and white_percentage >= 50: 
                copy_and_rename_image(thumbnail_path,path_out + "/clouds50_75/", str(count_files_in_folder(path_out + "/clouds50_75/"))) 
            if white_percentage > 75: 
                copy_and_rename_image(thumbnail_path,path_out + "/clouds75_100/", str(count_files_in_folder(path_out + "/clouds75_100/"))) 
        
        
        
def createLocations(path_out):
    create_folder(path_out + "train/clouds0_25")
    create_folder(path_out + "train/clouds25_50")
    create_folder(path_out + "train/clouds50_75")
    create_folder(path_out + "train/clouds75_100")
    create_folder(path_out + "test/clouds0_25")
    create_folder(path_out + "test/clouds25_50")
    create_folder(path_out + "test/clouds50_75")
    create_folder(path_out + "test/clouds75_100")

def convertTif2Jpg(path_out):
    convert_images(path_out + "train/clouds0_25")
    convert_images(path_out + "train/clouds25_50")
    convert_images(path_out + "train/clouds50_75")
    convert_images(path_out + "train/clouds75_100")
    convert_images(path_out + "test/clouds0_25")
    convert_images(path_out + "test/clouds25_50")
    convert_images(path_out + "test/clouds50_75")
    convert_images(path_out + "test/clouds75_100")
    
    
    