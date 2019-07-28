import os
import librosa
import numpy as np

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

def get_ref():
    folder = './../../saivt-bnews/The_Sydney_Morning_Herald_MRSS_Feed/'
    for fid in os.listdir(folder):
        f = folder + fid + '/' + fid + '.diarref.lab'
        #print(os.path.exists(f))
        lab2ref(f, fid)
        
    
def lab2ref(filepath, fid):
    print('converting ', fid)
    sol = []
    f = open(filepath, "r")
    for row in f:
        r = row.split("\t")
        if len(r) > 1:
            st = float(r[0])
            et = float(r[1])
            dt = np.around(et - st, decimals = 5)
            speaker = r[2].split(' ')[0].replace('\n','')
            #print(st, dt, speaker, r)
            #print(r)
            
            l = 'SPEAKER '+fid+' 1 '+str(st)+' '+str(dt)+' <NA> <NA> '+speaker+' <NA>\n'
            #print(l)
            sol.append(l)
    
    outf = open('./eval-tools/all'+'.rttm',"a+")    
    #outf = open('./eval-tools/ref/'+fid+'.rttm',"a")    
    outf.writelines(sol)
    #outf.write('\n')
    outf.close()
    
    f.close()

def CutFileByDuration(filepath, duration):
    f = open(filepath, "r")
    outf = open('./eval-tools/all_'+ str(duration) +'.rttm',"a+")
    
    sol = []
    ToWrite = False
    for row in f:
        r = row.split(' ')
        st = float(r[3])
        dt = float(r[4])
        et = st + dt
        if st >= duration:
            ToWrite = True
            break
        if et >= duration:
            r[4] = str(np.around(duration - st, 4))
            sol.append(' '.join(r))
            ToWrite = True
            break
        sol.append(row)
        
    if ToWrite:
        outf.writelines(sol)
    f.close()
    outf.close()

def CutAllFiles():
    folder = './eval-tools/ref/'
    for f in os.listdir(folder):
        #if '3054300' in f:
            print('cut ', f)
            CutFileByDuration(folder + f, 240)


def numberOfspks_offline():
    from collections import defaultdict
    folder = './eval-tools/ref/'
    d = defaultdict(list)
    
    for f in os.listdir(folder):
        fid = f.replace('.rttm','')
        num_spks = getNumberOfSpks(folder + f)
        if num_spks >= 5:
            d[5].append(fid)
        else:
            d[num_spks].append(fid)
            
    print('*'*10)
    
    new_folder = './out/offline/'
    for id in d.keys():
        outf = open('./out/offline/spks_'+ str(id) +'.rttm',"a+")
        print('writing spks = ', id)
        for fid in d[id]:
            filepath = new_folder + fid + '.rttm'

            f = open(filepath, "r")
            sol = []
            for row in f:
                sol.append(row)
            outf.writelines(sol)
            
        outf.close()


def numberOfspks_UISRNN():
    from collections import defaultdict
    folder = './eval-tools/ref/'
    d = defaultdict(list)
    
    for f in os.listdir(folder):
        fid = f.replace('.rttm','')
        num_spks = getNumberOfSpks(folder + f)
        if num_spks >= 5:
            d[5].append(fid)
        else:
            d[num_spks].append(fid)
            
    print('*'*10)
    
    new_folder  = './../Speaker-Diarization UISRNN/out/'
    for id in d.keys():
        outf = open('./out/uisrnn/spks_'+ str(id) +'.rttm',"a+")
        print('writing spks = ', id)
        for fid in d[id]:
            filepath = new_folder + fid + '.rttm'

            f = open(filepath, "r")
            sol = []
            for row in f:
                sol.append(row)
            outf.writelines(sol)
            
        outf.close()

def combineUISRNN():
    folder = './../Speaker-Diarization UISRNN/out/'
    outf = open('./out/uisrnn/all.rttm',"a+")
    
    n = 0
    for filename in os.listdir(folder):
        if 'spks' not in filename and 'all' not in filename:
            n+=1

            filepath = folder + filename
            print(filepath)

            f = open(filepath, "r")
            sol = []
            for row in f:
                sol.append(row)
            outf.writelines(sol)
            f.close()
            
    outf.close()
    print(n)
def combineOffline():
    folder = './out/offline/'
    outf = open('./out/offline/all.rttm',"a+")
    
    n = 0
    for filename in os.listdir(folder):
        if 'spks' not in filename and 'all' not in filename:
            n+=1

            filepath = folder + filename
            print(filepath)

            f = open(filepath, "r")
            sol = []
            for row in f:
                sol.append(row)
            outf.writelines(sol)
            f.close()
            
    outf.close()
    print(n)
    
def numberOfspks():
    from collections import defaultdict
    folder = './eval-tools/ref/'
    d = defaultdict(list)
    
    for f in os.listdir(folder):
        fid = f.replace('.rttm','')
        num_spks = getNumberOfSpks(folder + f)
        if num_spks >= 5:
            d[5].append(fid)
        else:
            d[num_spks].append(fid)
            
    print('*'*10)
    
    for id in d.keys():
        outf = open('./eval-tools/spks_'+ str(id) +'.rttm',"a+")
        print('writing spks = ', id)
        for fid in d[id]:
            filepath = folder + fid + '.rttm'
            f = open(filepath, "r")
            sol = []
            for row in f:
                sol.append(row)
            outf.writelines(sol)
            
        outf.close()

def getNumberOfSpks(filepath):
    f = open(filepath, "r")
    spks = []
    for row in f:
        r = row.split(' ')
        spk = r[7]
        if spk not in spks:
            spks.append(spk)
        
    
    f.close()
    print(r[1], spks)
    return len(spks)


def spectogram():
    
    import matplotlib.pyplot as plt
    from scipy import signal
    from scipy.io import wavfile
    import librosa
    
    signalData, samplingFrequency = librosa.load('./audio/3054300.wav',sr=None)    
    import matplotlib.pyplot as plot
        
    # Plot the signal read from wav file
    
    plot.subplot(211)
    plot.title('Spectrogram of a wav file with piano music')
    
    plot.plot(signalData)
    plot.xlabel('Sample')
    plot.ylabel('Amplitude')
    
    plot.subplot(212)
    
    plot.specgram(signalData,Fs=samplingFrequency)
    
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    
    plot.show()

if __name__ == "__main__":   
    #spectogram()
    numberOfspks_UISRNN()
    combineUISRNN()
    #get_ref()
    #folder = './audio/all/'
    #check_duration(folder)