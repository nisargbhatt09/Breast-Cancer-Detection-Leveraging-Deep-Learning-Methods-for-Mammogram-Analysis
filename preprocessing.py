import os
import pydicom
from PIL import Image
import numpy as np

def dicom_to_png(source_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(source_folder):
        if file.endswith('.dcm'):
            dicom_image = pydicom.dcmread(os.path.join(source_folder, file))
            pil_image = Image.fromarray(dicom_image.pixel_array)
            pil_image.save(os.path.join(output_folder, file.replace('.dcm', '.png')))

dicom_to_png('dicom_folder', 'png_folder')
