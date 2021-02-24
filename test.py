import keras
import numpy as np
import os
import glob
from skimage.io import imread
from skimage.transform import resize
import skimage
#import input_cbcr
import input
import without_MLF
import xception_model
import xception_agg
#import merge_model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

filename = 'test data'
log_path = 'log/'

batch_size = 30
epochs = 1200
img_cols = 128
img_row = 128
channel = 3
num_class = 2

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
                #img = skimage.io.imread(im)
                #img = resize(img, (128, 128))
                imgs.append(im)
                labels.append(idx)
            except:
                continue

    labels = np.asarray(labels, np.int32)
    labels = keras.utils.to_categorical(labels, num_class)


    return np.asarray(imgs), np.array(labels)



if __name__ == '__main__':

    imgs,label = read_img(filename,num_class)
    my_test_batch_generator = input.MY_Generator(imgs, label, batch_size)
    input_img = (img_cols,img_row,channel)


    model = xception_agg.Xception(img_cols=img_cols,img_rows=img_row,channel=channel,num_class=num_class)
    model.load_weights('the model file')

    model.summary()
    scores = model.evaluate_generator(generator=my_test_batch_generator,steps = len(imgs)/batch_size)
    print("test loss: ", scores[0])
    print("test acc: ",scores[1])