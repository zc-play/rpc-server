# coding: utf-8
import os
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, History

from .config import Config
from ..face_classifier.net import nn_base


cfg = Config()
epochs = 2000


def get_rec_model(pre_train_path, is_train=False):
    img_input = Input(shape=(64, 64, 3))
    x = nn_base(img_input, trainable=True)
    # decrease parameters numbers
    x = Conv2D(4096, (1, 1), activation='relu')(x)
    x = AveragePooling2D((4, 4), name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    # dense output
    x = Dense(cfg.class_size, activation='softmax', name='fc2622')(x)
    model = Model(inputs=img_input, outputs=x)
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.input_shape, 'trainable:{}'.format(layer.trainable))

    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    if is_train:
        model.load_weights(pre_train_path, by_name=True)
    return model


def train(train_path, val_path):
    pre_train_path = cfg.pre_train_model_path
    if os.path.exists(cfg.model_path):
        while True:
            is_import = input('The model already exists, is import it and continue to train(Y) or overwrite it(N): ')
            if is_import.lower() == 'y':
                pre_train_path = cfg.model_path
                break
            elif is_import.lower() == 'n':
                break
    model = get_rec_model(pre_train_path, is_train=True)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(64, 64),
        batch_size=10,
        seed=5
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(64, 64),
        batch_size=32,
        seed=5
    )

    model_checkpoint = ModelCheckpoint(cfg.model_path, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(patience=1)
    board = TensorBoard(cfg.train_log)
    hist = History()

    model.fit_generator(
        train_generator,
        steps_per_epoch=2,
        epochs=epochs,
        workers=8,
        use_multiprocessing=False,
        validation_data=None,
        callbacks=[model_checkpoint, reduce_lr, board, hist]
    )

    print('train completed!!!')


if __name__ == '__main__':
    # env PYTHONPATH='..' python rec_net.py
    train(cfg.train_path, cfg.dev_path)
