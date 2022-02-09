import numpy as np
import csv
import os
import librosa
from tqdm.auto import tqdm
import argparse



tqdmcols = 80
dataset_path = './data/gtzan'
features_path = f'{dataset_path}/features'
ignorefiles = ['jazz.00054.wav']



assert os.path.isdir(dataset_path), "[ERROR!] Can't find the dataset!"
if not os.path.isdir(features_path):
    os.makedirs(features_path)


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-s", "--usesplit",
                    action = "store_true",
                    help = "Use audio files split into 3 sec bits")

parser.add_argument("-n", "--numeric",
                    action = "store_true",
                    help = "Extract only numeric features, like mfcc and spectral_bandwidth")

parser.add_argument("-m", "--melspecs",
                    action = "store_true",
                    help = "Extract only mel-scaled spectrograms")

parser.add_argument("-a", "--allfeats",
                    action = "store_true",
                    help = "Extract both numeric features and melspectrograms")

# Read arguments from command line
args = parser.parse_args()

if args.usesplit:
    diraudiofiles = f'{dataset_path}/genres_3sec'
    csv_path = f'{dataset_path}/features/features_3sec.csv'
    melspec_path = f'{features_path}/melspecs_3sec'
else:
    diraudiofiles = f'{dataset_path}/genres_original'
    csv_path = f'{dataset_path}/features/features_30sec.csv'
    melspec_path = f'{features_path}/melspecs_30sec'



assert os.path.isdir(dataset_path), "[ERROR!] Can't find the dataset!"
if not os.path.isdir(melspec_path):
    os.makedirs(melspec_path)


genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
# genres = ['jazz']



def extract_numeric_features():
    print("Extracting numeric features...")
    return
    header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    data_arr = None
    for g in tqdm(genres, ncols=tqdmcols):
        for filename in tqdm(os.listdir(f'{diraudiofiles}/{g}'), leave=False, ncols=tqdmcols):
            filepath = f'{diraudiofiles}/{g}/{filename}'
            # print(filepath)
            if filename in ignorefiles:
                continue
            y, sr = librosa.load(filepath)
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



def extract_melspectograms():
    print("Extracting mel-scaled spectrograms...")
    return
    for g in genres:
        dirpath = os.path.join(f'{melspec_path}/{g}')
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        for filename in tqdm(os.listdir(f'{diraudiofiles}/{g}')):
            if filename in ignorefiles:
                continue
            filepath  =  f'{diraudiofiles}/{g}/{filename}'
            melspecimgfile = f'{melspec_path}/{g}/{filename.replace(".wav",".png")}'
            # print(melspecimgfile)
            if os.path.isfile(melspecimgfile):
                continue
            y,sr = librosa.load(filepath)
            mels = librosa.feature.melspectrogram(y=y,sr=sr)
            p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
            plt.savefig(melspecimgfile)

def extract_allfeatures():
    extract_numeric_features()
    extract_melspectograms()

def main():
    if args.numeric:
        extract_numeric_features()
    elif args.melspecs:
        extract_melspectograms()
    elif args.allfeats:
        extract_allfeatures()

if __name__ == '__main__':
    main()
