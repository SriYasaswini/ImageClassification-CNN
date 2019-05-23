#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy.misc import imresize

#matplotlib inline

import os

import tflearn
from tflearn.data_utils import image_preloader
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_augmentation import ImageAugmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import csv


"""
Change this if needed
"""

data_path = 'food8\\'



def pred_checker(model, path):
    test_path = "test/"
    for item in os.listdir(test_path):
        im = Image.open(open("test/" + item, 'rb'))
        im = im.resize((200, 200))
        pic = np.array(im)

        pic = pic.reshape(-1, 200, 200, 3)
        pred = model.predict(pic)

        i = int(np.where(np.array(pred[0]) == np.max(pred))[0])
        p = sorted([c[13:] for c in glob(path + 'images/*')])[i]

        plt.imshow(im)
        plt.show()
        print('Prediction: ', p)

        with open("nutrition_values.csv") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["product_name"] == p:
                    # print("Food Name: ", row['product_name'])
                    print("Energey/100g: ", row["energy_100g"])
                    print("Carbohydrates/100g: ", row["carbohydrates_100g"])
                    print("Sugar/100g: ", row["sugars_100g"])
                    print("Proteins/100g: ", row["proteins_100g"])
                    print("Fat/100g: ", row["fat_100g"])
                    print("Fiber/100g: ", row["fiber_100g"])
                    print("Cholestrol/100g: ", row["cholesterol_100g"] + "\n\n")

                    break

root_dir = 'food8\\images\\'
rows = 2
cols = 4
fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(15, 25))
fig.suptitle('Random Image from Each Food Class', fontsize=20)
sorted_food_dirs = sorted(os.listdir(root_dir))
for i in range(rows):
    for j in range(cols):
        try:
            food_dir = sorted_food_dirs[i*cols + j]
        except:
            break
        all_files = os.listdir(os.path.join(root_dir, food_dir))
        rand_img = np.random.choice(all_files)
        img = plt.imread(os.path.join(root_dir, food_dir, rand_img))
        ax[i][j].imshow(img)
        ec = (0, .6, .1)
        fc = (0, .7, .2)
        ax[i][j].text(0, -20, food_dir, size=10, rotation=0,
                ha="left", va="top",
                bbox=dict(boxstyle="round", ec=ec, fc=fc))
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



data_path = 'food8\\'


X, Y = image_preloader(data_path + 'images/', image_shape = (200, 200), mode = 'folder')

img_aug = ImageAugmentation()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, 200, 200, 3],
                    data_augmentation=img_aug)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 8, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0,
                   best_val_accuracy = 0.6)



## Comment the below two lines if model already exits
# model.fit(X, Y, n_epoch=5, shuffle=True,
#            batch_size=128,snapshot_epoch=True
#           )
#
# model.save('best_model8/model')


# Prediction of images:

model.load('best_model8/model')
pred_checker(model, data_path)



test_path = 'valid/'
X_valid, Y_valid = image_preloader(test_path + 'images/', image_shape = (200, 200), mode = 'folder')

results=model.predict(X_valid)
accuracy = model.evaluate(X_valid, Y_valid)[0] * 100.0
print("Accuracy: "+ str(accuracy))
