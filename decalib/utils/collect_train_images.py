import os, glob
import random
import numpy as np

files_dict_files = ['/mnt/lustre/wuxingzhe/vggface2/files_dict.npy', '/mnt/lustre/wuxingzhe/VoxCelebA2/files_dict.npy', \
                    '/mnt/lustre/wuxingzhe/BUPT-Balanced/African_files_dict.npy', '/mnt/lustre/wuxingzhe/BUPT-Balanced/Asian_files_dict.npy']
files_dict = {}
for files_dict_file in files_dict_files:
    files_dict.update(np.load(files_dict_file, allow_pickle = True).item())

img_files = []
for name_key in files_dict.keys():
    img_files.extend(files_dict[name_key])
random.shuffle(img_files)

dst_train_subject_images_file = './train_images.txt'
w1 = open(dst_train_subject_images_file, 'w')
for img_file in img_files[:2000000]:
    w1.write(img_file+'\n')
w1.close()

