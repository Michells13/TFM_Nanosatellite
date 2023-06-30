import cv2
import os

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

# Example usagew
folder_path = "C:/Users/MICHE/Documents/Datasets/Cloudsen12/split/train/clouds0_25"
# folder_path1 = "C:/Users/MICHE/Documents/Datasets/Cloudsen12/split/train/clouds25_50"
# folder_path2 = "C:/Users/MICHE/Documents/Datasets/Cloudsen12/split/train/clouds50_75"
# folder_path3 = "C:/Users/MICHE/Documents/Datasets/Cloudsen12/split/train/clouds75_100"
convert_images(folder_path)
# convert_images(folder_path1)
# convert_images(folder_path2)
# convert_images(folder_path3)










