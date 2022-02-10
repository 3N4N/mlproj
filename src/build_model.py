import os
import random
import shutil
from tqdm.auto import tqdm
import numpy as np

from keras import layers, models
from keras import optimizers
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator




# reset this variable if you haven't split the dataset already
train_test_split_done = True

tqdmcols = 80
dataset_path = './data/gtzan'
features_path = f'{dataset_path}/features'
audios_path = f'{dataset_path}/genres_3sec'
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
        filenames = os.listdir(os.path.join(f'{audios_path}/{g}'))
        random.shuffle(filenames)
        test_files = filenames[:100]
        train_files = filenames[100:]
        for f in tqdm(train_files, leave=False, ncols=tqdmcols):
            shutil.copy(f'{audios_path}/{g}/{f}', f'{train_dir}/{g}')
        for f in tqdm(test_files, leave=False, ncols=tqdmcols):
            shutil.copy(f'{audios_path}/{g}/{f}', f'{test_dir}/{g}')



def gen_img_data(target_size=(640,480)):
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=target_size,
                                                        color_mode="rgba",
                                                        class_mode='categorical',
                                                        batch_size=128)

    validation_dir = test_dir
    vali_datagen = ImageDataGenerator(rescale=1./255)
    vali_generator = vali_datagen.flow_from_directory(validation_dir,
                                                      target_size=target_size,
                                                      color_mode='rgba',
                                                      class_mode='categorical',
                                                      batch_size=128)


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

    model = Model(inputs=X_input,outputs=X,name='GenreModel')
    return model




if __name__ == "__main__":
    if not train_test_split_done:
        prep_train_test()
