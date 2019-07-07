import os, sys, glob
import configparser
from diarizationFunctions import *
import numpy as np
sys.path.append('visualization')
from viewer2 import PlotDiar

import time
import librosa

def runDiarization(showName, config):   
    
    wav_path = config['PATH']['audio']+showName
        
    y_total, sr = librosa.load(wav_path,sr=None)
    audio_duration = librosa.get_duration(y_total, sr=sr)
    
    framelength = config.getfloat('FEATURES','framelength')
    frameshift = config.getfloat('FEATURES','frameshift')
    nfilters = config.getint('FEATURES','nfilters')
    ncoeff = config.getint('FEATURES','ncoeff')
    
    
    frame_length_inSample=framelength*sr
    hop = int(frameshift*sr)
    NFFT=int(2**np.ceil(np.log2(frame_length_inSample)))
    
    seg_length = config.getint('SEGMENT','length')
    seg_incre = config.getint('SEGMENT','increment')
    seg_rate = config.getint('SEGMENT','rate')
    
    KBM_window_length = config.getint('KBM','windowLength')
    KBM_minG = config.getint('KBM','minimumNumberOfInitialGaussians')
    KBM_minW = config.getint('KBM','maximumKBMWindowRate')
    KBM_rel_size = config.getfloat('KBM','relKBMsize')
    
    bk_bits = config.getfloat('BINARY_KEY','bitsPerSegmentFactor')
    init_cluster = config.getint('CLUSTERING','N_init')
    metric = config['CLUSTERING']['metric']
    

    
    step = 1
    i = step
    y = [] 
    
    while i < audio_duration:
        if i + step >= audio_duration:
            y = y_total
        else:
            y = y_total[0 : i * sr]
            

        i = i + step
    
        
    
        t0 = time.time()  
        
        
        allData = extractFeaturesFromSignal(y, sr, nfilters,ncoeff, NFFT, hop)
        nFeatures = allData.shape[0]    
        
        t1 = time.time()
        feature_t = t1 - t0
        
        maskUEM = np.ones([1,nFeatures])     
        maskSAD = getSADfromSignal(y, sr, frameshift, nFeatures)
        
        t2 = time.time()
        SAD_t = t2 - t1
        mask = np.logical_and(maskUEM, maskSAD)    
        mask = mask[0][0:nFeatures]
        
        nSpeechFeatures=np.sum(mask)
        speechMapping = np.zeros(nFeatures)
        #you need to start the mapping from 1 and end it in the actual number of features independently of the indexing style
        #so that we don't lose features on the way
        speechMapping[np.nonzero(mask)] = np.arange(1,nSpeechFeatures+1)
        data=allData[np.where(mask==1)]
        del allData        
        
        segmentTable=getSegmentTable(mask,speechMapping, seg_length, seg_incre, seg_rate)
        numberOfSegments=np.size(segmentTable,0)
        #create the KBM
        #set the window rate in order to obtain "minimumNumberOfInitialGaussians" gaussians
        
        try:
            if np.floor((nSpeechFeatures-KBM_window_length)/KBM_minG) < KBM_minW:
                windowRate = int(np.floor((np.size(data,0)-KBM_window_length)/KBM_minG))
            else:
                windowRate = KBM_minW     
          
            poolSize = np.floor((nSpeechFeatures-KBM_window_length)/windowRate)
    
            kbmSize = int(np.floor(poolSize*KBM_rel_size))
            kbm, gmPool = trainKBM(data,KBM_window_length,windowRate,kbmSize )      
            Vg = getVgMatrix(data,gmPool,kbm,config.getint('BINARY_KEY','topGaussiansPerFrame')) 
        except: 
            print('time: ', i, ' Failed: windowRate: ', windowRate, ' poolSize: ', poolSize, ' kbm size: ', kbmSize)
            continue
        
        
        #print('second: ',  i,  ' data: ',data.shape, ' KBM: ', kbmSize)

        
        
        segmentBKTable, segmentCVTable = getSegmentBKs(segmentTable, kbmSize, Vg, bk_bits, speechMapping)
        
        #print(i,np.size(data,0),nSpeechFeatures,np.size(segmentTable,0),np.size(Vg,0),segmentCVTable.shape,kbmSize)
            
        
        #print('time:', i , 'data:', data.shape,'Vg:', Vg.shape, 'segT:', segmentTable.shape, 
        #      'CVT:', segmentCVTable.shape, 'kbm:', kbmSize)    
        
        t3 = time.time()
        KBM_t = t3 - t2

        
        initialClustering = np.digitize(np.arange(numberOfSegments),np.arange(0,numberOfSegments,numberOfSegments/init_cluster))
        finalClusteringTable, k = performClustering(speechMapping, segmentTable, segmentBKTable, segmentCVTable, Vg, bk_bits, kbmSize, init_cluster, initialClustering, metric)        
        bestClusteringID = getBestClustering(metric, segmentBKTable, segmentCVTable, finalClusteringTable, k)
        
        t4 = time.time()
        clustering_t = t4 - t3

        
        finalClusteringTableResegmentation,finalSegmentTable = performResegmentation(data,speechMapping, mask,finalClusteringTable[:,bestClusteringID.astype(int)-1],segmentTable,config.getint('RESEGMENTATION','modelSize'),config.getint('RESEGMENTATION','nbIter'),config.getint('RESEGMENTATION','smoothWin'),nSpeechFeatures)
        t5 = time.time()
        reseg_t = t5 - t4
        tu = t5 - t0
        #getSegmentationFile(config['OUTPUT']['format'],frameshift,finalSegmentTable, np.squeeze(finalClusteringTableResegmentation), showName, config['EXPERIMENT']['name'], config['PATH']['output'], config['EXTENSION']['output'])
        #print(feature_t, SAD_t, KBM_t, clustering_t, reseg_t, tu)
        #wav_path = './audio_test/2.wav'
        #print(config['PATH']['audio']+showName+'.wav')
        
        print('time:', i , 'data:', data.shape,'Vg:', Vg.shape, 'segT:', segmentTable.shape, 
              'CVT:', segmentCVTable.shape, 'kbm:', kbmSize, 'C:',finalClusteringTable[:,bestClusteringID.astype(int)-1].shape,'finalC:',finalClusteringTableResegmentation.shape,'finalSeg:',finalSegmentTable.shape)    
        
        
        speakerSlice = getSegResultForPlot(frameshift,finalSegmentTable, np.squeeze(finalClusteringTableResegmentation))
        
        #getSegResultForPlotlater(frameshift,finalSegmentTable, np.squeeze(finalClusteringTableResegmentation), showName, i, tu)
        #print('time: ', i, 'used:', tu)
       
        '''
        if i % 10 == 0:
            print('plot: ', i)
            p = PlotDiar(map=speakerSlice, wav=wav_path, title = 'Binary key diarization: ' +wav_path   +', number of speakers: ' + str(len(speakerSlice)), gui=True, pick=True, size=(25, 6))
            p.draw()
            p.plot.show()
        
        '''
    
    #getSegmentationFile(config['OUTPUT']['format'],config.getfloat('FEATURES','frameshift'),segmentTable, finalClusteringTable[:,bestClusteringID.astype(int)-1], showName, config['EXPERIMENT']['name'], config['PATH']['output'], config['EXTENSION']['output'])      
    

        
    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

def plotRealtimeResult(showName):
    wav_path = config['PATH']['audio']+showName
    result_path = './out/' + showName.replace('.wav', '') + '.rttm'
    speakerSlice = readSegResultforPlot(result_path)
    
    y, sr = librosa.load(wav_path,sr=None)
    duration = librosa.get_duration(y, sr=sr)
    
    p = PlotDiar(map=speakerSlice, wav=wav_path, duration = duration+5, title = 'Binary key diarization: ' +wav_path   +', number of speakers: ', gui=True)
    wm = p.plot.get_current_fig_manager()
    wm.window.state('zoomed')
    p.plot.show()

if __name__ == "__main__":     
    # If a config file in INI format is passed by argument line then it's used. 
    # For INI config formatting please refer to https://docs.python.org/3/library/configparser.html
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]
    else:
        configFile = 'config-test.ini'    
    config = configparser.ConfigParser()
    config.read(configFile)
    
    
    filename = '1.wav'
    #filename = '3055877.wav'
    #filename = '3057402.wav'
    #filename = 'chinese_same.wav'
            
    runDiarization(filename, config)
    #plotRealtimeResult(filename)