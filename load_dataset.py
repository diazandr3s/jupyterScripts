'''
Data loader
Author: Andres Diaz-Pinto
'''

import numpy as np
import glob
from PIL import Image
import cv2 as cv
from random import shuffle

class load_dataset:

    def __init__(self, root_data, image_width, image_height, channel, batch_size):
        self.img_rows = image_width
        self.img_cols = image_height
        self.channel = channel
        self.batch_size = batch_size
        self.filelistTrain = glob.glob(root_data)
        shuffle(self.filelistTrain)
        load_dataset.names = []

    def load_img(self, i):

        if '_g_' in self.filelistTrain[i]:
            label = 1
        else:
            label = 0

        #print(self.filelistTrain[i])
        img = Image.open(self.filelistTrain[i]).convert('RGB')
        img = img.resize([self.img_rows, self.img_cols])
        img = np.array(img)
        img = np.array(cv.normalize(img.astype(np.float32), np.zeros((self.img_rows, self.img_cols, self.channel), dtype=np.float32), -1, 1, cv.NORM_MINMAX))

        return img, label

    def data2predict(self):

        ini = 0

        all_imgs = len(self.filelistTrain)

        while ini <= all_imgs:

            batch_imgs = []

            if (ini + self.batch_size) < all_imgs:

                for i in range(ini, ini + self.batch_size):

                    image_sample, _ = self.load_img(i)
                    load_dataset.names.append(self.filelistTrain[i])
                    batch_imgs.append(image_sample)

                batch_imgs = np.array(batch_imgs).astype(np.float32)

                ini += self.batch_size

            else:

                for i in range(ini, all_imgs):
                    image_sample, _ = self.load_img(i)
                    load_dataset.names.append(self.filelistTrain[i])
                    batch_imgs.append(image_sample)

                batch_imgs = np.array(batch_imgs).astype(np.float32)

                ini = all_imgs


            yield batch_imgs

