import os, glob
import random
import numpy as np

id_names = []
id_names_files = ['/mnt/lustre/wuxingzhe/vggface2/id_names.npy', '/mnt/lustre/wuxingzhe/VoxCelebA2/id_names.npy']
img_paths = ['/mnt/lustre/wuxingzhe/vggface2/crop256', '/mnt/lustre/wuxingzhe/VoxCelebA2/images/color']

for i,id_names_file in enumerate(id_names_files):
    names = np.load(id_names_file)
    paths = [os.path.join(img_paths[i], name) for name in names]
    id_names.extend(paths)
random.shuffle(id_names)

files_dict_files = ['/mnt/lustre/wuxingzhe/vggface2/files_dict.npy', '/mnt/lustre/wuxingzhe/VoxCelebA2/files_dict.npy']
files_dict = {}
for files_dict_file in files_dict_files:
    files_dict.update(np.load(files_dict_file, allow_pickle = True).item())

nums_dict_files = ['/mnt/lustre/wuxingzhe/vggface2/num_dict.npy', '/mnt/lustre/wuxingzhe/VoxCelebA2/num_dict.npy']
nums_dict = {}
for nums_dict_file in nums_dict_files:
    nums_dict.update(np.load(nums_dict_file, allow_pickle = True).item())

dst_train_subject_images_file = './train_detail_subject_images.txt'
w1 = open(dst_train_subject_images_file, 'w')

top = 0; num_per_person = 6; person_num = 4
files_buffer = []; pos_buffer = np.zeros((person_num)).astype(np.int32); len_buffer = np.zeros((person_num)).astype(np.int32)
for i in range(person_num):
    files_buffer.append(files_dict[id_names[i]])
    len_buffer[i] = nums_dict[id_names[i]]
top = person_num
is_full = False

while True:
    for i in range(person_num):
        if len_buffer[i] == pos_buffer[i]:
            if top >= len(id_names):
                is_full = True
                break

            files_buffer[i] = files_dict[id_names[top]]
            len_buffer[i] = nums_dict[id_names[top]]
            for j in range(num_per_person):
                w1.write(files_buffer[i][j]+'\n')
            pos_buffer[i] = num_per_person
            top += 1

        elif len_buffer[i] - pos_buffer[i] < num_per_person:
            for j in range(pos_buffer[i], len_buffer[i]):
                w1.write(files_buffer[i][j]+'\n')
            for j in range(num_per_person + pos_buffer[i] - len_buffer[i]):
                w1.write(files_buffer[i][j]+'\n')
            pos_buffer[i] = len_buffer[i]
        else:
            for j in range(pos_buffer[i], pos_buffer[i]+num_per_person):
                w1.write(files_buffer[i][j]+'\n')
            pos_buffer[i] += num_per_person

    if is_full:
        break
w1.close()
