import os
import random
import shutil
from tqdm.auto import tqdm
import numpy as np


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



if __name__ == "__main__":
    if not train_test_split_done:
        prep_train_test()
