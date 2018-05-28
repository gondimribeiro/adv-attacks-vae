import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
import os

img_list = []
count_imgs = [0] * 3
with open('./data/celeba/list_eval_partition.txt', 'r') as fp:
    for line in fp:
        img_name, img_set = line.split(' ')
        img_set = int(img_set)
        count_imgs[img_set] = count_imgs[img_set] + 1
        img_list.append([img_name, img_set])

imgs_dir = './data/celeba/img_align_celeba/'

data_mean = 0.431751299266
data_std = 0.300219581459

train_x = np.memmap('.tmp_celeba_train.npy', np.float32, 'w+',
                    shape=(count_imgs[0] + count_imgs[1], 64, 64, 3))
test_x = np.memmap('.tmp_celeba_test.npy', np.float32, 'w+',
                   shape=(count_imgs[2], 64, 64, 3))

train_count = 0
test_count = 0
for i in img_list:
    img_name, img_set = i
    img = skimage.transform.resize(plt.imread(imgs_dir + img_name), (64, 64))
    img = (img - data_mean) / data_std

    if img_set == 2:
        test_x[test_count] = img
        test_count = test_count + 1
    else:
        train_x[train_count] = img
        train_count = train_count + 1

np.random.shuffle(train_x)

np.save('./data/celeba/celeba_64x64_train_normalized', train_x)
np.save('./data/celeba/celeba_64x64_test_normalized', test_x)

os.remove(".tmp_celeba_train.npy")
os.remove(".tmp_celeba_test.npy")
