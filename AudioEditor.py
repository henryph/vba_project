import librosa
import wave
import pydub

def trim_audio(path, duration):
    y, sr = librosa.load(path,sr=None,duration=duration)
    
    new_path = './audio_test/2_' + str(duration) + '.wav'
    print(new_path)
    librosa.output.write_wav(new_path, y, sr)
    
    #
    # wf = wave.open(new_path, 'rb')

def trim_audio_2(path, duration):
    from pydub import AudioSegment
    t = duration * 1000
    newaudio = AudioSegment.from_wav(path)
    newaudio = newaudio[:t]
    new_path = './audio_test/2_' + str(duration) + '.wav'
    newaudio.export(new_path, format = "wav")
    
if __name__ == "__main__": 
    for t in range(0, 300, 20):
        print(t)
        trim_audio_2('./audio_test/EN2005_test.wav', t)
    #trim_audio_2('./audio_test/2.wav', 90)