import librosa
from diarizationFunctions import *
import time
import numpy as np
import configparser

def diarizaton_rt_simulate(filepath, config):
    y, sr = librosa.load(filepath,sr=None)
    #y = y[0:102*sr-1]
    print(len(y))
    duration = len(y) / sr
    duration_int = int(np.floor(duration))+1

    framelength = 0.025
    
    frameshift = 0.01
    nfilters = 30
    ncoeff = 30
    
    step = 5
    
    frame_length_inSample=framelength*sr
    hop = int(frameshift*sr)
    NFFT=int(2**np.ceil(np.log2(frame_length_inSample)))
    
    
    t = time.time()
    Features = np.empty((0,ncoeff))
    
    i = 0
    while i < duration_int:
        newData = y[i*sr: (i+step) * sr-1]
        f = extractFeaturesFromSignal(newData, sr, nfilters,ncoeff, NFFT, hop)
        Features = np.concatenate((Features, f), axis = 0)
        print(i, f.shape, Features.shape)
        
        
        
        
        i = i+step
        
    print(time.time()-t)
    
    print(Features.shape)
    
    t = time.time()
    allFeatures = extractFeaturesFromSignal(y, sr, nfilters,ncoeff, NFFT, hop)
    print(time.time()-t)
    
    
    print(allFeatures.shape)
    #print((Features == allFeatures).all())
    #print(sum(Features-allFeatures))
    nFeatures = allFeatures.shape[0]  
    
    
    
    maskSAD1 = getSADfromSignal(y, sr, frameshift, nFeatures)
    maskSAD2 = getSADfile(config,'2',nFeatures)
    print(maskSAD1.shape)
    print(maskSAD2.shape)
    print((maskSAD1 == maskSAD2).all())

    
if __name__ == "__main__":  
    configFile = 'config-test.ini'    
    config = configparser.ConfigParser()
    config.read(configFile)
    
    diarizaton_rt_simulate('./audio_test/2.wav', config)