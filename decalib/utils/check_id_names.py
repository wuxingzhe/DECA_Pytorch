import os
train_id_names = []

r1 = open('train_subject_images.txt')
lines = r1.readlines()
r1.close()
for line in lines:
    line = line.strip()
    if 'vggface2' in line:
        train_id_names.append(line.split('/')[-2])

r1 = open('train_images.txt')
lines = r1.readlines()
r1.close()
for line in lines:
    line = line.strip()
    if 'vggface2' in line:
        train_id_names.append(line.split('/')[-2])
train_id_names = list(set(train_id_names))

id_names = os.listdir('/mnt/lustre/wuxingzhe/vggface2/crop256')
test_id_names = list(set(train_id_names).difference(set(id_names)))
print(str(len(test_id_names)))
