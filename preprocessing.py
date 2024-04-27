import os
import pydicom
from PIL import Image
import numpy as np

classes = ['cancer', 'non_cancer']


def dcm_to_png(source_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(source_folder):
        if file.endswith('.dcm'):
            dicom_image = pydicom.dcmread(os.path.join(source_folder, file))
            pil_image = Image.fromarray(dicom_image.pixel_array)
            pil_image.save(os.path.join(output_folder, file.replace('.dcm', '.png')))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def windowing():
    png_dir = os.path.expanduser('/kaggle/input/rsna-png-dataset/kaggle/working/cancer/')
    
    window_level = 127
    window_width = 255
    
    png_files = glob.glob(os.path.join(png_dir, '*.png'))

    for file in png_files:
    # Read the PNG file
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    # Perform windowing
    min_window = window_level - (window_width / 2)
    max_window = window_level + (window_width / 2)
    img_windowed = np.clip(img, min_window, max_window)
    img_windowed = (img_windowed - min_window) / window_width * 255
    
    # Save the windowed image as a new PNG file
    cv2.imwrite(f'{os.path.splitext(file)[0]}_windowed.png', img_windowed)

def hist_eq(src_dir, tar_dir):
    # png_dir = os.path.expanduser('/kaggle/input/rsna-png-dataset/kaggle/working/cancer/')
    # enhanced_dir = os.path.expanduser('/kaggle/working/enhanced_cancer_img')
    png_files = glob.glob(os.path.join(png_dir, '*.png'))

    for file in png_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        
        img_eq = cv2.equalizeHist(img)
        
        min_val = img_eq.min()
        max_val = img_eq.max()
        img_stretched = (img_eq - min_val) / (max_val - min_val) * 255
        
        filename = os.path.basename(file)
        cv2.imwrite(os.path.join(enhanced_dir, f'{os.path.splitext(filename)[0]}_enhanced.png'), img_stretched)
    
    
dcm_to_png('dicom_folder', 'png_folder')
