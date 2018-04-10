# coding: utf-8
import os
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, History

from ..face_classifier.config import Config

seed = 7
np.random.seed(seed)
conf = Config()
epochs = 2000


def conv2d_bn(input_tensor, filters, kernel_size, activation='relu',
              strides=(1, 1), padding='valid', name=None, trainable=True):
    x = Conv2D(filters, kernel_size, strides=strides,
               padding=padding, name=name, trainable=trainable)(input_tensor)
    x = BatchNormalization(trainable=trainable, name=name+'_bn')(x)
    x = Activation(activation)(x)
    return x


def nn_base(input_tensor=None, trainable=False):
    # Determine proper input shape
    input_shape = (None, None, 3)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = conv2d_bn(img_input, 64, (3, 3), padding='same', name='block1_conv1', trainable=trainable)
    x = conv2d_bn(x, 64, (3, 3), padding='same', name='block1_conv2', trainable=trainable)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv2d_bn(x, 128, (3, 3), padding='same', name='block2_conv1', trainable=trainable)
    x = conv2d_bn(x, 128, (3, 3), padding='same', name='block2_conv2', trainable=trainable)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv2d_bn(x, 256, (3, 3), padding='same', name='block3_conv1', trainable=trainable)
    x = conv2d_bn(x, 256, (3, 3), padding='same', name='block3_conv2', trainable=trainable)
    x = conv2d_bn(x, 256, (3, 3), padding='same', name='block3_conv3', trainable=trainable)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv2d_bn(x, 512, (3, 3), padding='same', name='block4_conv1', trainable=trainable)
    x = conv2d_bn(x, 512, (3, 3), padding='same', name='block4_conv2', trainable=trainable)
    x = conv2d_bn(x, 512, (3, 3), padding='same', name='block4_conv3', trainable=trainable)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # # Block 5  remove
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # x = GlobalMaxPooling2D()(x)

    return x


def get_classifier_model(is_train=False, pre_train_path=None):
    img_input = Input(shape=(64, 64, 3))
    x = nn_base(img_input, trainable=True)
    # fc1
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = BatchNormalization(trainable=True, name='fc1_bn')(x)
    x = Dropout(0.8, name='fc1_dropout')(x)
    # fc2
    x = Dense(256, activation='relu', name='fc2')(x)
    x = BatchNormalization(trainable=True, name='fc2_bn')(x)
    x = Dropout(0.9, name='fc2_dropout')(x)
    # dense output
    x = Dense(2, activation='softmax', name='dense_out')(x)
    model = Model(inputs=img_input, outputs=x)
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.input_shape, 'trainable:{}'.format(layer.trainable))
    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    if is_train:
        model.load_weights(pre_train_path, by_name=True)
    return model


def train(train_path, val_path):
    pre_train_path = conf.keras_vgg16_path
    if os.path.exists(conf.face_model_path):
        while True:
            is_import = input('The model already exists, is import it and continue to train(Y) or overwrite it(N): ')
            if is_import.lower() == 'y':
                pre_train_path = conf.face_model_path
                break
            elif is_import.lower() == 'n':
                break
    model = get_classifier_model(is_train=True, pre_train_path=pre_train_path)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(64, 64),
        batch_size=500,
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(64, 64),
        batch_size=200,
    )

    model_checkpoint = ModelCheckpoint(conf.face_model_path, save_best_only=False)
    reduce_lr = ReduceLROnPlateau(patience=1)
    board = TensorBoard(conf.train_log)
    hist = History()

    model.fit_generator(
        train_generator,
        steps_per_epoch=300,
        epochs=epochs,
        workers=8,
        use_multiprocessing=True,
        validation_data=val_generator,
        callbacks=[model_checkpoint, reduce_lr, board, hist]
    )

    print('train completed!!!')


def test(test_path, model_path):
    model = get_classifier_model()
    model.load_weights(model_path)
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(64, 64),
        batch_size=200,
    )
    hist = model.evaluate_generator(
        test_generator,
        steps=100,
        workers=4,
        use_multiprocessing=True
    )
    print(hist)


if __name__ == '__main__':
    # env PYTHONPATH='..' python net.py
    #train(conf.train_path, conf.dev_path)

    test(conf.test_path, conf.face_model_path)
