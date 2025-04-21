import os
import shutil
import cv2
import numpy as np
import re

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
            'channel': channel
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

def get_target_folder_by_magnification(source_folder_name):
    """Determine the target folder based on the magnification in the source folder name"""
    if "40xD" in source_folder_name:
        output_dir = "New_Nucleus/40x/40x-bright/output0"
    elif "20xD" in source_folder_name:
        output_dir = "New_Nucleus/20x/20x-bright/output0"
    elif "20xP" in source_folder_name:
        output_dir = "New_Nucleus/20x/20x-phase/output0"
    else:
        return None
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def copy_and_convert_tif_files(source_folder, target_folder, folder_name):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 获取前缀
    prefix = get_prefix_from_parent(source_folder)
    
    # 确定允许的 well 编号
    allowed_wells = set()
    if "VS-P013" in folder_name:
        allowed_wells = {"0008", "0009", "0014"}
    elif "VS-P014" in folder_name:
        allowed_wells = {"0008", "0009", "0014"}
    elif "VS-P015" in folder_name:
        allowed_wells = {"0008", "0009", "0014"}
    elif "VS-P017" in folder_name:
        allowed_wells = {"0008", "0009", "0014"}
    
    count = 0  # 统计转换的文件数
    input_file_count = len([f for f in os.listdir(source_folder) if f.endswith("C4.tif")])  # 统计输入文件数量

    for filename in os.listdir(source_folder):
        if filename.endswith("C4.tif"):
            file_info = parse_filename(filename)
            if file_info and file_info['well'] in allowed_wells:
                source_path = os.path.join(source_folder, filename)
                
                # 解析文件名并创建新的文件名
                new_filename = f"{prefix}_W{file_info['well'].zfill(4)}F{file_info['field'].zfill(4)}.tif"
                target_path = os.path.join(target_folder, new_filename)
                
                # 读取16位图像
                image = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
                if image is not None and image.dtype == np.uint16:
                    # 归一化到8位
                    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imwrite(target_path, image_8bit)
                    print(f"Converted and copied: {filename} -> {new_filename}")
                else:
                    shutil.copy2(source_path, target_path)
                    print(f"Copied: {filename} -> {new_filename}")
                
                count += 1  # 增加计数

    print(f"Total input files for {folder_name}: {input_file_count}")  # 输出该文件夹的输入文件数
    print(f"Total processed files for {folder_name}: {count}\n")  # 输出该文件夹处理的文件数

    return input_file_count, count  # 返回输入和处理文件数量

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

input_counts = {}  # 统计每个输入文件夹的文件数
processed_counts = {
    "New_Nucleus/40x/40x-bright/output0": 0,
    "New_Nucleus/20x/20x-bright/output0": 0,
    "New_Nucleus/20x/20x-phase/output0": 0
}

# Loop through each source folder and perform the conversion
for folder in folder_names:
    source_folder = f"../../../../mnt/nas/{folder}/Projection"
    target_folder = get_target_folder_by_magnification(folder)
    if target_folder:
        input_count, processed_count = copy_and_convert_tif_files(source_folder, target_folder, folder)
        processed_counts[target_folder] = len(os.listdir(target_folder))  # 统计输出文件夹
        input_counts[folder] = input_count  # 记录输入文件数量

# 输出最终统计结果
print("\nFinal Summary:")
print("Input Files per Folder:")
for folder, count in input_counts.items():
    print(f"{folder}: {count} files")

print("\nProcessed Files in Output Directories:")
for folder, count in processed_counts.items():
    print(f"{folder}: {count} files processed.")

# ls /home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-bright/input | wc -l
# ls /home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-bright/output0 | wc -l
# ls /home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-phase/input | wc -l
# ls /home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-phase/output0 | wc -l
# ls /home/yuqi/virtual_staining/tmp_data/Nucleus/40x/40x-bright/input | wc -l
# ls /home/yuqi/virtual_staining/tmp_data/Nucleus/40x/40x-bright/output0 | wc -l