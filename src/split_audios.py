import os


orig_dir = './data/gtzan/genres_original'
split_dir = './data/gtzan/genres_3sec'

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    path_audio = os.path.join(f'{split_dir}/{g}')
    if not os.path.isdir(path_audio):
        os.makedirs(path_audio)
    else:
        print(f'{path_audio} directory already exists or is a file.')




from pydub import AudioSegment
from tqdm.auto import tqdm

for g in tqdm(genres):
    j = -1
    for filename in tqdm(os.listdir(os.path.join(f"{orig_dir}/{g}")), leave=False):
        if filename == 'jazz.00054.wav':
            continue
        audiofile = f'{orig_dir}/{g}/{filename}'
        j += 1
        for w in range(0,10):
            t1 = 3*(w)*1000
            t2 = 3*(w+1)*1000
            newfileloc = f'{split_dir}/{g}/{g}.{format(int(str(j)+str(w)),"05d")}.wav'
            # print(newfileloc)
            if os.path.isfile(newfileloc):
                continue
            newAudio = AudioSegment.from_wav(audiofile)
            new = newAudio[t1:t2]
            new.export(newfileloc, format="wav")
