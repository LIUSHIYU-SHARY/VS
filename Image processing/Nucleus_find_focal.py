import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import re

def preprocess_image(img):
    """Preprocess an image by converting it to grayscale"""
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def focus_measure(img, method='brenner'):
    """Calculate the focus measure of an image using the specified method"""
    if method == 'brenner':
        diff_x = img[:, 2:] - img[:, :-2]
        diff_y = img[2:, :] - img[:-2, :]
        sum_squares = np.sum(diff_x ** 2) + np.sum(diff_y ** 2)
        return sum_squares / img.size
    elif method == 'laplacian':
        lap = cv2.Laplacian(img, cv2.CV_64F)
        return np.var(lap)
    elif method == 'tenengrad':
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        return np.mean(np.sqrt(gx**2 + gy**2))
    else:
        raise ValueError(f"Unsupported method: {method}")

def parse_filename(filename):
    """Parse a filename to extract metadata"""
    pattern = r'W(\d+)F(\d+)T(\d+)Z(\d+)C(\d+)\.tif'
    match = re.match(pattern, filename)
    if match:
        well, field, timestamp, zstack, channel = match.groups()
        return {
            'well': well,
            'field': field,
            'timestamp': timestamp,
            'zstack': zstack,
            'channel': channel,
            'group_key': f"W{well}F{field}T{timestamp}"
        }
    return None

def get_prefix_from_parent(parent_dir):
    """Extract the prefix from the parent directory name"""
    try:
        full_path = os.path.abspath(parent_dir)
        parent_folder = os.path.basename(os.path.dirname(full_path))
        prefix = parent_folder[2:12]
        return prefix
    except Exception as e:
        print(f"Warning: Could not extract prefix from parent folder: {str(e)}")
        return ""

def get_output_directory(folder_name):
    if "20xD" in folder_name:
        output_dir = "New_Nucleus/20x/20x-bright/input"
    elif "20xP" in folder_name:
        output_dir = "New_Nucleus/20x/20x-phase/input"
    elif "40xD" in folder_name:
        output_dir = "New_Nucleus/40x/40x-bright/input"
    else:
        return None
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def process_folder(input_dir, folder_name, method='brenner'):
    """Process a single folder of images with additional filtering based on folder name"""
    tif_files = sorted([f for f in os.listdir(input_dir) if f.endswith('C5.tif')])
    allowed_wells = set()
    if "VS-P013" in folder_name:
        allowed_wells = {"0008", "0009", "0014"}
    elif "VS-P014" in folder_name:
        allowed_wells = {"0008", "0009", "0014"}
    elif "VS-P015" in folder_name:
        allowed_wells = {"0008", "0009", "0014"}
    elif "VS-P017" in folder_name:
        allowed_wells = {"0008", "0009", "0014"}
    sample_groups = {}
    for file in tif_files:
        file_info = parse_filename(file)
        if file_info and file_info['channel'] == '5' and file_info['well'] in allowed_wells:
            group_key = file_info['group_key']
            if group_key not in sample_groups:
                sample_groups[group_key] = []
            sample_groups[group_key].append((file, input_dir))
    return sample_groups

def find_best_focus_multi_folders(parent_directory, folder_name, output_directory, method='brenner'):
    """Process all well folders in the parent directory"""
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    prefix = get_prefix_from_parent(parent_directory)
    well_folders = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d)) and d.lower().startswith('well')]
    all_groups = {}
    for folder in well_folders:
        folder_path = os.path.join(parent_directory, folder)
        groups = process_folder(folder_path, folder_name, method)
        all_groups.update(groups)
    print(f"Found {len(all_groups)} total groups with C5 images across {len(well_folders)} wells")
    for group_key, files in tqdm(all_groups.items()):
        try:
            scores = []
            images = []
            for file, source_dir in files:
                img_path = os.path.join(source_dir, file)
                img = cv2.imread(img_path, -1)
                if img is None:
                    print(f"Warning: Could not read {file}")
                    continue
                processed_img = preprocess_image(img)
                score = focus_measure(processed_img, method)
                scores.append(score)
                images.append((img, file))
            if not scores:
                print(f"No valid images found for group {group_key}")
                continue
            best_idx = np.argmax(scores)
            best_img, best_file = images[best_idx]
            file_info = parse_filename(best_file)
            if file_info:
                new_filename = f"{prefix}_W{file_info['well'].zfill(4)}F{file_info['field'].zfill(4)}.tif"
                output_path = os.path.join(output_directory, new_filename)
                image_8bit = cv2.normalize(best_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(output_path, image_8bit)
                print(f"Saved best focus image for {group_key}: {new_filename}")
        except Exception as e:
            print(f"Error processing group {group_key}: {str(e)}")
            continue

if __name__ == "__main__":
    folder_names = [
        "20250406T112114_VS-P013-20250403_20xD-3.0x13-5Ch",
        "20250407T170813_VS-P014-20250403_20xD-3.0x11-5Ch",
        "20250408T180227_VS-P015-20250403_20xD-3.0x10-5Ch",
        "20250410T113844_VS-P017-20250403_20xD-3.0x10-5Ch",
        "20250411T132340_VS-P013-20250403_20xP-4.0x07-5Ch",
        "20250411T220437_VS-P014-20250403_40xD-1.5x08-5Ch",
        "20250412T140235_VS-P015-20250403_20xP-3.0x06-5Ch",
        "20250412T174901_VS-P015-20250403_40xD-1.5x08-5Ch",
        "20250413T141301_VS-P017-20250403_20xP-3.0x07-5Ch",
        "20250413T175538_VS-P017-20250403_40xD-1.5x08-5Ch"
        # "20250107T181804_VS-P005-20241212_20xD-1.0x31-3Ch",
        # "20250107T183331_VS-P005-20241212_20xD-1.0x31-3Ch",
        # "20250107T184457_VS-P005-20241212_20xD-1.0x31-3Ch"
        
    ]
    
    parent_directory_base = "../../../../mnt/nas"
            
    for folder in folder_names:
        parent_directory = os.path.join(parent_directory_base, folder, "Image")
        output_directory = get_output_directory(folder)
        if output_directory:
            print(f"Processing {folder} -> {output_directory}")
            find_best_focus_multi_folders(parent_directory, folder, output_directory, method='tenengrad')
        else:
            print(f"Skipping {folder}, no matching output directory.")

 



