import keras
from keras.models import Model
from keras.layers import Input,Dense,PReLU,Dropout
import xception_agg

def model_merge(img_row, img_cols, channel):

    input_shape = (img_row, img_cols, channel)
    input_img1 = Input(shape=input_shape)
    input_img2 = Input(shape=input_shape)

    model1 = xception_agg.Xception(input_img1)
    model1.load()

    model2 = xception_agg.Xception(input_img2)
    model2.load()

    r1=model1.output
    r2=model2.output

    x = keras.layers.average([r1,r2])

    model = Model(input=[input_img1, input_img2], outputs=x)

    return model
