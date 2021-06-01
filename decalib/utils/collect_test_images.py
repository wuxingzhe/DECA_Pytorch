import os, glob

w1 = open('qualitative_test_images.txt', 'w')

src_path = '/mnt/lustre/wuxingzhe/DECA_Pytorch/TestSamples/examples'
for img_file in glob.glob(os.path.join(src_path, '*png')):
    w1.write(img_file+'\n')
for img_file in glob.glob(os.path.join(src_path, '*jpg')):
    w1.write(img_file+'\n')
w1.close()
