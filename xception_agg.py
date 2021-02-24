from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
import keras
from keras.regularizers import l2
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda
import keras.backend as K


def cbam_block(cbam_feature, ratio=16):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)  # 通道维度
    cbam_feature = spatial_attention(cbam_feature)  # 空间维度
    return cbam_feature


# 通道维度
def channel_attention(input_feature, ratio=8):
    # 获取当前的维度顺序

    channel = input_feature._keras_shape[-1]

    # shareMLP-W0
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    # shareMLP-W1
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    # 经过全局平均池化
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)  # [1:]分片操作，shape（1,1,1,channel）——>（1,1,channel）
    # assert断言操作，如果为false抛出异常

    # 经过W1
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    # 经过W2
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    # 经过全局最大池化
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    # 经过W1
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    # 经过W2
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    # 逐元素相加，sigmoid激活
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    # 逐元素相乘，得到下一步空间操作的输入feature
    return multiply([input_feature, cbam_feature])


# 空间维度
def spatial_attention(input_feature):
    kernel_size = 7  # 卷积核7x7

    channel = input_feature._keras_shape[-1]  # 取最后一个元素（channel）
    cbam_feature = input_feature

    # 经过平均池化
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    # axis等于几，就理解成对那一维值进行压缩；二维矩阵（0为列1为行），x为四维矩阵（1,h,w,channel）所以axis=3，对矩阵每个元素求平均；
    # keepdims保持其矩阵二维特性
    assert avg_pool._keras_shape[-1] == 1  # 检验channel压缩为1

    # 经过最大池化
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1

    # 从channel维度进行拼接
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2  # 检验channel为2

    # 进行卷积操作
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    # 逐元素相乘，得到feature
    return multiply([input_feature, cbam_feature])


def Xception(img_rows, img_cols, channel, num_class):
    input_shape = (img_rows, img_cols, channel)
    input_img = Input(shape=input_shape)

    # block 1
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization()(x)
    cbam_block(x,16)
    x = Activation('relu')(x)
    x_1 = x



    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # block 2
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = cbam_block(x, 16)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x_2 = x

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # block 3
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = cbam_block(x, 16)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x_3 = x

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)


    # block 4
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = cbam_block(x, 16)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x_4 = x

    x_concat_1 = keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_1)
    x_concat_12 = keras.layers.concatenate([x_concat_1, x_2], axis=-1)
    x_concat_12_res = keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_concat_12)
    x_concat_123 = keras.layers.concatenate([x_concat_12_res, x_3], axis=-1)
    x_concat_123_res = keras.layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x_concat_123)
    x_concat = keras.layers.concatenate([x_concat_123_res, x_4], axis=-1)
    #x_concat = Conv2D(2048, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x_concat)

    x_concat = GlobalAveragePooling2D()(x_concat)



    # 循环
    for i in range(8):
        residual = x

        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = cbam_block(x, 16)

        x = layers.add([x, residual])

    # ========EXIT FLOW============
    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = cbam_block(x, 16)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False,name='separable')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x, 16)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D(name='globave')(x)
    #x = keras.layers.concatenate([x, x_concat], axis=-1)  #0
    #x = layers.add([x, x_concat])
    #x = keras.layers.Dropout(0.5)(x)
    x = Dense(2048, activation='relu', kernel_regularizer=l2(0.0001))(x)
    #x = layers.add([x, x_concat])
    x = keras.layers.concatenate([x, x_concat], axis=-1)        #1
    x = keras.layers.Dropout(1)(x)                       #1
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = keras.layers.Dropout(1)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0001))(x)
    #x = BatchNormalization()(x)  #0
    x = keras.layers.Dropout(1)(x)
    x = Dense(num_class,kernel_initializer='he_normal',activation='softmax')(x)
    model = Model(input_img, x, name='xception_agg')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.00001, decay=0.00001), metrics=['accuracy'])
    return model
    '''
    itializer='he_normal', activation='sigmoid')(x)

    # create model
    model = Model(input_img, x, name='xception')

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.00002, decay=0.00001),
                  metrics=['accuracy'])

    return model


'''




