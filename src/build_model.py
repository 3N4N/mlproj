import os
import random
import shutil
from tqdm.auto import tqdm
import numpy as np

import tensorflow as tf
from keras import layers, models
# from keras import optimizers
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator




# reset this variable if you haven't split the dataset already
train_test_split_done = True

tqdmcols = 80
dataset_path = './data/gtzan'
features_path = f'{dataset_path}/features'
melspec_path = f'{dataset_path}/melspecs_3sec'
csv_path = f'{dataset_path}/features/features_3sec.csv'
melspec_path = f'{dataset_path}/melspecs_3sec'
ignorefiles = ['jazz.00054.wav']


genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

train_dir = f'{dataset_path}/train'
test_dir = f'{dataset_path}/test'

if not os.path.isdir(train_dir):
    os.makedirs(train_dir)
if not os.path.isdir(test_dir):
    os.makedirs(test_dir)

def prep_train_test():
    for g in tqdm(genres, ncols=tqdmcols):
        if not os.path.isdir(f'{train_dir}/{g}'):
            os.makedirs(f'{train_dir}/{g}')
        if not os.path.isdir(f'{test_dir}/{g}'):
            os.makedirs(f'{test_dir}/{g}')
        filenames = os.listdir(os.path.join(f'{melspec_path}/{g}'))
        random.shuffle(filenames)
        train_files = filenames[:900]
        test_files = filenames[900:]
        for f in tqdm(train_files, leave=False, ncols=tqdmcols):
            shutil.copy(f'{melspec_path}/{g}/{f}', f'{train_dir}/{g}')
        for f in tqdm(test_files, leave=False, ncols=tqdmcols):
            shutil.copy(f'{melspec_path}/{g}/{f}', f'{test_dir}/{g}')



def gen_img_data(train_dir, validation_dir, target_size=(640,480)):
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=target_size,
                                                        color_mode="rgba",
                                                        class_mode='categorical',
                                                        batch_size=128)

    validation_dir = test_dir
    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_generator = valid_datagen.flow_from_directory(validation_dir,
                                                      target_size=target_size,
                                                      color_mode='rgba',
                                                      class_mode='categorical',
                                                      batch_size=128)
    return train_generator, valid_generator


def model_genre(input_shape = (640,480,4), classes = 10):
    np.random.seed(9)
    X_input = layers.Input(input_shape)

    X = layers.Conv2D(8, kernel_size=(3,3), strides=(1,1),
                      kernel_initializer=glorot_uniform(seed=9))(X_input)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((2,2))(X)

    X = layers.Conv2D(16, kernel_size=(3,3), strides=(1,1),
                      kernel_initializer=glorot_uniform(seed=9))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((2,2))(X)

    X = layers.Conv2D(32, kernel_size=(3,3), strides = (1,1),
                      kernel_initializer = glorot_uniform(seed=9))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((2,2))(X)

    X = layers.Conv2D(64, kernel_size=(3,3), strides=(1,1),
                      kernel_initializer=glorot_uniform(seed=9))(X)
    X = layers.BatchNormalization(axis=-1)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((2,2))(X)


    X = layers.Flatten()(X)
    X = layers.Dense(classes, activation='softmax', name='fc' + str(classes),
                     kernel_initializer = glorot_uniform(seed=9))(X)

    model = models.Model(inputs=X_input,outputs=X,name='GenreModel')
    return model



import keras.backend as K
def get_f1(y_true, y_pred): # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

if __name__ == "__main__":
    if not train_test_split_done:
        prep_train_test()

    train_generator, valid_generator = gen_img_data(train_dir, test_dir, target_size=(640,480))

    model = model_genre(input_shape=(640,480,4), classes=10)
    opt = tf.keras.optimizers.Adam(learning_rate = 5e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', get_f1])
    model.summary()

    model.fit(train_generator, epochs=40, validation_data=valid_generator)

    preds = model.evaluate(x=X_test,y=Y_test)
    print(preds[1])
    print(preds[2])
