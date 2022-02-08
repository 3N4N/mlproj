import numpy as np
import pandas as pd
import csv
import os
import librosa
from tqdm.auto import tqdm


tqdmcols = 80
dataset_path = './input/gtzan/'
csv_path = './data.csv'

df = pd.read_csv(dataset_path + 'features_3_sec.csv')
df.columns

header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
genres = ['jazz']

diraudiofiles = dataset_path + "genres_original/"

file = open(csv_path, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

data_arr = None
for g in tqdm(genres, ncols=tqdmcols):
    for filename in tqdm(os.listdir(diraudiofiles + g), leave=False, ncols=tqdmcols):
        songname = diraudiofiles + f'{g}/{filename}'
        if filename == 'jazz.00054.wav':
            continue
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        tarr = np.array([filename, np.mean(chroma_stft), np.mean(rms),
                         np.mean(spec_cent), np.mean(spec_bw),
                         np.mean(rolloff), np.mean(zcr)]).astype(object)
        if data_arr is None:
            data_arr = tarr
        else:
            data_arr = np.vstack([data_arr, tarr])

np.savetxt(csv_path, data_arr, delimiter=',', header=','.join(header), comments='', fmt="%s")
