import keras
from input import MY_Generator
import input
import numpy as np
import os
import xception_agg
import xception_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

filename = 'train data'
filename_val = 'val data'
#model_path = 'model/cbam_qxcep_lr.hd5'
log_path = 'log/'
batch_size = 32
epochs = 1200
img_cols = 128
img_row = 128
channel = 3
num_class = 2

def get_batch_data():
    images, label = input.read_img(filename,num_class)
    # 打乱图片顺序
    num_example = images.shape[0]
    #print(num_example)
    arr = np.arange(num_example)
    #print(arr)
    np.random.shuffle(arr)
    data = images[arr]
    label = label[arr]

    # 将所有数据分为训练集和验证集
    ratio = 0.9
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]


    return x_train,y_train,x_val,y_val


if __name__ == '__main__':

    x_train,y_train,x_val,y_val= get_batch_data()

    my_training_batch_generator = MY_Generator(x_train,y_train,batch_size)
    my_validation_batch_generator = MY_Generator(x_val,y_val,batch_size)

    input_img = (img_cols,img_row,channel)
    print("RGB..................")
    #model = xception_model.Xception(img_cols,img_row,channel,num_class)
    model = xception_agg.Xception(img_row,img_cols,channel,num_class)
    model.summary()

    ck = keras.callbacks.ModelCheckpoint(
        '/'+'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.hd5',
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1
    )
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=3, verbose=1, mode='auto',
                                        cooldown=0, min_lr=0)
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_path, write_images=True, histogram_freq=0)
    cbks = [ck,lr,tb_cb]

    hist = model.fit_generator(generator=my_training_batch_generator,
                        steps_per_epoch=(len(x_train) // batch_size),
                        epochs=epochs,
                        verbose=1,
                        validation_data=my_validation_batch_generator,
                        validation_steps=(len(x_val) // batch_size),
                        callbacks=cbks
                        )
