import tensorflow as tf
import os
import glob
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import keras
import cv2
from scipy import ndimage


def read_img(path,num_class):
    filename = os.listdir(path)
    filename.sort(key=lambda x: int(x[0]))
    cate = [path + x for x in filename]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*'):
            print('reading the images:%s' % (im))
            try:
                #img = io.imread(im)
                #img = transform.resize(img, (128, 128))
                imgs.append(im)
                labels.append(idx)
            except:
                continue

    labels = np.asarray(labels, np.int32)
    labels = keras.utils.to_categorical(labels, num_class)


    return np.asarray(imgs), labels

class MY_Generator(keras.utils.Sequence): # generator 继承自 Sequence

    def __init__(self, image_filenames, labels, batch_size):
        # image_filenames - 图片路径
        # labels - 图片对应的类别标签
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        # 计算 generator要生成的 batches 数，
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))   # 这里别忘了int()转化成整数

    def __getitem__(self, idx):
        kernel_3x3 = np.array([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]])
        # idx - 给定的 batch 数，以构建 batch 数据 [images_batch, GT]
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]


        x_RGB = [
            resize(imread(file_name), (128, 128))
               for file_name in batch_x]
        x_RGB = np.asarray(x_RGB, np.float32)

        return x_RGB, batch_y
        #x_Ycbcr = np.asarray([cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) for img in x_RGB])
        #x_Ycbcr = np.asarray(x_Ycbcr,np.float32)
        # x_cbcr = []
        # for img in x_Ycbcr:
        #     Y,Cr,Cb = cv2.split(img)
        #     #img_concat = np.expand_dims(Y,-1)
        #     Cr = (Cr-np.min(Cr))/(np.max(Cr)-np.min(Cr))
        #     Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
        #     #img_concat = cv2.merge([Cr, Cb])
        #     Cb = (Cb - np.min(Cb)) / (np.max(Cb) - np.min(Cb))
        #     #img_concat = np.expand_dims(Y,-1);
        #     img_concat = cv2.merge([Y,Cr,Cb])
        #     x_cbcr.append(img_concat)
        # x_cbcr = np.asarray(x_cbcr)

        #return x_Ycbcr/255.0,batch_y



        #x_HSV = np.asarray([cv2.cvtColor(img,cv2.COLOR_BGR2HSV) for img in x_RGB])
        #return x_HSV/255.0, batch_y
        '''
        x_YCrCb = np.asarray([cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) for img in x_RGB])
        img_new_hs = []
        img_new_cbcr = []


        for img in x_HSV:
            H,S,V = cv2.split(img)
            H = ndimage.convolve(H, kernel_3x3)
            S = ndimage.convolve(S, kernel_3x3)
            V = ndimage.convolve(V, kernel_3x3)
            img_concat = cv2.merge([H,S,V])
            img_new_hs.append(img_concat)

        for img in x_RGB:
            B,G,R = cv2.split(img)
            B = ndimage.convolve(B, kernel_3x3)
            G = ndimage.convolve(G, kernel_3x3)
            R = ndimage.convolve(R, kernel_3x3)
            img_concat = cv2.merge([B,G,R])
            img_new_cbcr.append(img_concat)
        img_new_cbcr = np.asarray(img_new_cbcr, np.float32)
        img_new_hs = np.asarray(img_new_hs, np.float32)


        #img_new = np.asarray(img_new)
        return [img_new_cbcr/255.0,img_new_hs/255.0], batch_y
      '''