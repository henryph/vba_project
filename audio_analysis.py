import os
import librosa

def check_duration(folder):
    l = [0 for j in range(30)]
    for filename in os.listdir(folder):
        wav_path = folder + filename
        y, sr = librosa.load(wav_path,sr=None)
        duration = librosa.get_duration(y, sr=sr)
        n = int(duration / 20)
        for i in range(n):
            l[i] += 1
        print (filename, duration)
    
    print(l)

def split_ref():
    pass


if __name__ == "__main__":   
    folder = './audio/all/'
    check_duration(folder)