import os
import librosa
from tqdm.auto import tqdm
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt


tqdmcols = 80

dataset_path = './data/gtzan'
audio_dir = f'{dataset_path}/genres_test'
test_dir = f'{dataset_path}/test1'
if not os.path.isdir(test_dir):
    os.makedirs(test_dir)

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()


def extract_melspectograms(audio_dir, melspec_path):
    print("Extracting mel-scaled spectrograms...")
    plt.switch_backend('agg')
    for filename in tqdm(os.listdir(f'{audio_dir}'), ncols=tqdmcols):
        if ".wav" not in filename:
            continue
        filepath  =  f'{audio_dir}/{filename}'
        melspecimgfile = f'{melspec_path}/{filename.replace(".wav",".png")}'
        # print(melspecimgfile)

        # y,sr = librosa.load(filepath, offset=40, duration=3)
        # mels = librosa.feature.melspectrogram(y=y,sr=sr)
        # p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
        # plt.axis('off')
        # plt.savefig(melspecimgfile)
        # plt.close('all')

        dur = 3
        totdur = int(librosa.get_duration(filename=filepath))
        for t1 in range(10,totdur,dur):
            melspecimgfile = f'{melspec_path}/{filename.replace(".wav",f".{t1}.png")}'
            y,sr = librosa.load(filepath, offset=t1, duration=dur)
            mels = librosa.feature.melspectrogram(y=y,sr=sr)
            p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
            plt.axis('off')
            plt.savefig(melspecimgfile)
            plt.close('all')



def main():
    extract_melspectograms(audio_dir, test_dir)

if __name__ == '__main__':
    main()
