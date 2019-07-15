import os, sys, glob
import configparser
from diarizationFunctions import *
import numpy as np
sys.path.append('visualization')
from viewer import PlotDiar

import time
import librosa

def runDiarization(showName, config):   

    wav_path = './audio_runtime/'+showName
        
    y, sr = librosa.load(wav_path,sr=None)
    audio_duration = librosa.get_duration(y, sr=sr)
    
    
    framelength = config.getfloat('FEATURES','framelength')
    frameshift = config.getfloat('FEATURES','frameshift')
    nfilters = config.getint('FEATURES','nfilters')
    ncoeff = config.getint('FEATURES','ncoeff')
    
    frame_length_inSample=framelength*sr
    hop = int(frameshift*sr)
    NFFT=int(2**np.ceil(np.log2(frame_length_inSample)))
     
    print('Filename\t\t',showName)
    print('Extracting features')  
    
    t0 = time.time()  
    allData = extractFeaturesFromSignal(y, sr, nfilters,ncoeff, NFFT, hop)        
    nFeatures = allData.shape[0]    
    
    t1 = time.time()
    feature_t = t1 - t0
    
    

    maskUEM = np.ones([1,nFeatures])     
    maskSAD = getSADfromSignal(y, sr, frameshift, nFeatures)     
   
    t2 = time.time()
    SAD_t = t2 - t1
    
    mask = np.logical_and(maskUEM,maskSAD)    
    mask = mask[0][0:nFeatures]
    

    nSpeechFeatures=np.sum(mask)
    speechMapping = np.zeros(nFeatures)
    speechMapping[np.nonzero(mask)] = np.arange(1,nSpeechFeatures+1)
    data=allData[np.where(mask==1)]
    del allData        
    
    segmentTable=getSegmentTable(mask,speechMapping,config.getint('SEGMENT','length'),config.getint('SEGMENT','increment'),config.getint('SEGMENT','rate'))
    numberOfSegments=np.size(segmentTable,0)

    t3 = time.time()
    seg_t = t3 - t2
    
    #create the KBM
    #set the window rate in order to obtain "minimumNumberOfInitialGaussians" gaussians
    
    if np.floor((nSpeechFeatures-config.getint('KBM','windowLength'))/config.getint('KBM','minimumNumberOfInitialGaussians')) < config.getint('KBM','maximumKBMWindowRate'):
        windowRate = int(np.floor((np.size(data,0)-config.getint('KBM','windowLength'))/config.getint('KBM','minimumNumberOfInitialGaussians')))
    else:
        windowRate = int(config.getint('KBM','maximumKBMWindowRate'))        
    poolSize = np.floor((nSpeechFeatures-config.getint('KBM','windowLength'))/windowRate)
    if  config.getint('KBM','useRelativeKBMsize'):
        kbmSize = int(np.floor(poolSize*config.getfloat('KBM','relKBMsize')))
    else:
        kbmSize = int(config.getint('KBM','kbmSize'))        
    kbm, gmPool = trainKBM(data,config.getint('KBM','windowLength'),windowRate,kbmSize )  
    
    t4 = time.time()
    KBM_t = t4 - t3

    Vg = getVgMatrix(data,gmPool,kbm,config.getint('BINARY_KEY','topGaussiansPerFrame'))    

    segmentBKTable, segmentCVTable = getSegmentBKs(segmentTable, kbmSize, Vg, config.getfloat('BINARY_KEY','bitsPerSegmentFactor'), speechMapping)    
    
    t5 = time.time()
    BKCV_t = t5 - t4
    
    
    initialClustering = np.digitize(np.arange(numberOfSegments),np.arange(0,numberOfSegments,numberOfSegments/config.getint('CLUSTERING','N_init')))
    if config.getint('CLUSTERING','linkage'):
        finalClusteringTable, k = performClusteringLinkage(segmentBKTable, segmentCVTable, config.getint('CLUSTERING','N_init'), config['CLUSTERING']['linkageCriterion'], config['CLUSTERING']['metric'])
    else:
        finalClusteringTable, k = performClustering(speechMapping, segmentTable, segmentBKTable, segmentCVTable, Vg, config.getfloat('BINARY_KEY','bitsPerSegmentFactor'), kbmSize, config.getint('CLUSTERING','N_init'), initialClustering, config['CLUSTERING']['metric'])        
    if config['CLUSTERING_SELECTION']['bestClusteringCriterion'] == 'elbow':
        bestClusteringID = getBestClustering(config['CLUSTERING_SELECTION']['metric_clusteringSelection'], segmentBKTable, segmentCVTable, finalClusteringTable, k)
    elif config['CLUSTERING_SELECTION']['bestClusteringCriterion'] == 'spectral':
        bestClusteringID = getSpectralClustering(config['CLUSTERING_SELECTION']['metric_clusteringSelection'],finalClusteringTable,config.getint('CLUSTERING','N_init'),segmentBKTable,segmentCVTable,k,config.getint('CLUSTERING_SELECTION','sigma'),config.getint('CLUSTERING_SELECTION','percentile'),config.getint('CLUSTERING_SELECTION','maxNrSpeakers'))+1        

    t6 = time.time()
    clustering_t = t6 - t5
    
    if config.getint('RESEGMENTATION','resegmentation') and np.size(np.unique(finalClusteringTable[:,bestClusteringID.astype(int)-1]),0)>1:
        finalClusteringTableResegmentation,finalSegmentTable = performResegmentation(data,speechMapping, mask,finalClusteringTable[:,bestClusteringID.astype(int)-1],segmentTable,config.getint('RESEGMENTATION','modelSize'),config.getint('RESEGMENTATION','nbIter'),config.getint('RESEGMENTATION','smoothWin'),nSpeechFeatures)

        t7 = time.time()
        reseg_t = t7 - t6
        #getSegmentationFile(config['OUTPUT']['format'],config.getfloat('FEATURES','frameshift'),finalSegmentTable, np.squeeze(finalClusteringTableResegmentation), showName, config['EXPERIMENT']['name'], config['PATH']['output'], config['EXTENSION']['output'])

        

        '''
        speakerSlice = getSegResultForPlot(config['OUTPUT']['format'],config.getfloat('FEATURES','frameshift'),finalSegmentTable, np.squeeze(finalClusteringTableResegmentation), showName, config['EXPERIMENT']['name'], config['PATH']['output'], config['EXTENSION']['output'])
        p = PlotDiar(map=speakerSlice, wav=wav_path, title = 'Binary key diarization: ' +wav_path   +', number of speakers: ' + str(len(speakerSlice)), gui=True, pick=True, size=(25, 6))
        p.draw()
        p.plot.show()
        '''
        
    else:
        reseg_t = 0
        pass
        #getSegmentationFile(config['OUTPUT']['format'],config.getfloat('FEATURES','frameshift'),segmentTable, finalClusteringTable[:,bestClusteringID.astype(int)-1], showName, config['EXPERIMENT']['name'], config['PATH']['output'], config['EXTENSION']['output'])      
    
    tu = time.time() - t0
    
    time_list = ['%.5f' % t for t in [feature_t, 
                                      SAD_t, 
                                      seg_t, 
                                      KBM_t, 
                                      BKCV_t, 
                                      clustering_t, 
                                      reseg_t, 
                                      tu, 
                                      audio_duration, 
                                      feature_t / audio_duration,
                                      SAD_t / audio_duration,
                                      seg_t / audio_duration,
                                      KBM_t / audio_duration,
                                      BKCV_t / audio_duration,
                                      clustering_t / audio_duration,
                                      reseg_t / audio_duration,
                                      tu/audio_duration]]
    
    time_list = [showName] + time_list
    return time_list
    
    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
if __name__ == "__main__":     
    # If a config file in INI format is passed by argument line then it's used. 
    # For INI config formatting please refer to https://docs.python.org/3/library/configparser.html
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]
    else:
        configFile = 'config-test.ini'    
    config = configparser.ConfigParser()
    config.read(configFile)
  
    import csv
    import os
    
    from os import system, name 

    # define our clear function 
    def clear(): 
  
        # for windows 
        if name == 'nt': 
            _ = system('cls') 
      
    
    folder = './audio_runtime/'
    print(folder)
    
    
    
    with open('runtime_stat.csv', 'w', newline='') as csvfile:
        twriter = csv.writer(csvfile, delimiter=',')
        twriter.writerow(['filename','feature_t', 'SAD_t', 'seg_t', 'KBM_t','BKCV_t', 'clustering_t', 'reseg_t', 
                          'total', 'duration', 
                          'feature_RF', 'SAD_RF', 'seg_RF', 'KBM_RF', 'BKCV_RF', 'clustering_RF', 'reseg_RF'
                          'RF'])
    
        # Files are diarized one by one
        for filename in os.listdir(folder):
            #if os.path.basename(filename) == "":
                all_timelist = []
                all_totalruntime = []
                for i in range(0, 5):
                    l = runDiarization(filename, config)
                    all_timelist.append(l)
                    all_totalruntime.append(l[8])
                
                medIdx = all_timelist.index(np.percentile(a,50,interpolation='nearest'))
                print(all_timelist, ' : ', medIdx, '-->', all_timelist[medIdx])
                twriter.writerow(all_timelist[medIdx])

                clear()
                
