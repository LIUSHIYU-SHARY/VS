import os

def find_duplicate_filenames(folder1, folder2, output_file):
    # 获取两个文件夹中的文件名列表
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # 查找相同的文件名
    duplicates = files1.intersection(files2)

    if duplicates:
        # 如果有重复的文件名，保存到txt文件
        with open(output_file, 'w') as f:
            for filename in duplicates:
                f.write(filename + '\n')
        print(f"找到 {len(duplicates)} 个重复的文件名，已保存到 {output_file}")
    else:
        # 如果没有重复的文件名，输出“无重复”
        print("无重复")

# # 示例用法
# folder1 = '/home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-bright/test/a'
# folder2 = '/home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-bright/train/a'
# folder1 = '/home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-phase/test/a'
# folder2 = '/home/yuqi/virtual_staining/tmp_data/Nucleus/20x/20x-phase/train/a'
folder1 = '/home/yuqi/virtual_staining/tmp_data/Nucleus/40x/40x-bright/test/a'
folder2 = '/home/yuqi/virtual_staining/tmp_data/Nucleus/40x/40x-bright/train/a'

output_file = 'duplicates-Nucleus-20x-bright.txt'

find_duplicate_filenames(folder1, folder2, output_file)