# coding: utf-8
import time
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.utils import Sequence
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


def load_model(name):
    model = None

    if name == 'res50':
        model = ResNet50(weights='imagenet', include_top=False)
    elif name == 'vgg16':
        model = VGG16(weights='imagenet', include_top=False)
    elif name == 'vgg19':
        model = VGG19(weights='imagenet')
    elif name == 'inception':
        model = InceptionV3(weights='imagenet')

    return model



def predict():
    model = load_model('res50')
    img_path = '/data/algo/data/dataset/lfw/Ana_Claudia_Talancon/Ana_Claudia_Talancon_0001.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    beg_time = time.time()
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    # print('Predicted:', decode_predictions(preds, top=3)[0])
    print(preds.shape)
    print('The total time: %s' % (time.time() - beg_time))


def resnet_fine_tune():
    # create the base pre-train model
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    # new model to train
    model = Model(inputs=base_model.input, outputs=preds)

    # train only the top layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adma', loss='categorical_crossentropy')
    train_datagen = ImageDataGenerator(featurewise_center=True,
                       featurewise_std_normalization=True,
                       rotation_range=20,
                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory('/data/',
                                                        target_size=(227, 227),
                                                        batch_size=32)

    validation_generator = test_datagen.flow_from_directory('/data/',
                                                            target_size=(227, 227),
                                                            batch_size=32)
    model.fit_generator(train_generator, epochs=10, validation_data=validation_generator, workers=4)





if __name__ == '__main__':
    predict()
