import glob, os

r1 = open('train_subject_images.txt')
lines = r1.readlines()
r1.close()
all_lines = [line.strip() for line in lines]
r1 = open('train_images.txt')
lines = r1.readlines()
r1.close()
all_lines.extend([line.strip() for line in lines])

id_names = ['n009270', 'n000008', 'n000011', 'n000012', 'n009275', 'n009278']
root_path = '/mnt/lustre/wuxingzhe/vggface2/crop256'

for name in id_names:
    img_path = os.path.join(root_path, name)
    img_files = glob.glob(os.path.join(img_path, '*jpg'))
    for img_file in img_files:
        if img_file not in all_lines:
            print(img_file)
