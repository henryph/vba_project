# AUTHORS
# Jose PATINO, EURECOM, Sophia-Antipolis, France, 2019
# http://www.eurecom.fr/en/people/patino-jose
# Contact: patino[at]eurecom[dot]fr, josempatinovillar[at]gmail[dot]com

import os, sys, glob
import configparser
from diarizationFunctions import *
import numpy as np
sys.path.append('visualization')
from viewer import PlotDiar

import time
import librosa

def runDiarization(showName, config):   
    
    wav_path = config['PATH']['audio']+showName+'.wav'
        
    y, sr = librosa.load(wav_path,sr=None)
    audio_duration = librosa.get_duration(y, sr=sr)
    
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
    
    # check same:
    allData = extractFeaturesFromSignal(y, sr, nfilters,ncoeff, NFFT, hop)
    allData2 = extractFeatures(wav_path,framelength,frameshift,nfilters,ncoeff)    
    print((allData == allData2).all())
    nFeatures = allData.shape[0]   
    
    maskSAD = getSADfromSignal(y, sr, frameshift, nFeatures)
    maskSAD2 = getSADfile(config,'2',nFeatures)
    print((maskSAD == maskSAD2).all())
    
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
    if np.floor((nSpeechFeatures-KBM_window_length)/KBM_minG) < KBM_minW:
        windowRate = int(np.floor((np.size(data,0)-KBM_window_length)/KBM_minG))
    else:
        windowRate = KBM_minW     
      
    poolSize = np.floor((nSpeechFeatures-KBM_window_length)/windowRate)
    
    kbmSize = int(np.floor(poolSize*KBM_rel_size))
    kbm, gmPool = trainKBM(data,KBM_window_length,windowRate,kbmSize )      

    Vg = getVgMatrix(data,gmPool,kbm,config.getint('BINARY_KEY','topGaussiansPerFrame'))    
    segmentBKTable, segmentCVTable = getSegmentBKs(segmentTable, kbmSize, Vg, bk_bits, speechMapping)    

    t3 = time.time()
    KBM_t = t3 - t2
    #print("Time used for traing KBM and cal BK, CV: ", KBM_t)
    
    initialClustering = np.digitize(np.arange(numberOfSegments),np.arange(0,numberOfSegments,numberOfSegments/init_cluster))
    finalClusteringTable, k = performClustering(speechMapping, segmentTable, segmentBKTable, segmentCVTable, Vg, bk_bits, kbmSize, init_cluster, initialClustering, metric)        
    bestClusteringID = getBestClustering(metric, segmentBKTable, segmentCVTable, finalClusteringTable, k)
    
    t4 = time.time()
    clustering_t = t4 - t3
    #print("Time used for clustering: ",clustering_t)
    

    finalClusteringTableResegmentation,finalSegmentTable = performResegmentation(data,speechMapping, mask,finalClusteringTable[:,bestClusteringID.astype(int)-1],segmentTable,config.getint('RESEGMENTATION','modelSize'),config.getint('RESEGMENTATION','nbIter'),config.getint('RESEGMENTATION','smoothWin'),nSpeechFeatures)

    t5 = time.time()
    reseg_t = t5 - t4
    print("Time used for resegmentation: ",  reseg_t)

    tu = t5 - t0
    print('Total time used:', tu)



    #getSegmentationFile(config['OUTPUT']['format'],frameshift,finalSegmentTable, np.squeeze(finalClusteringTableResegmentation), showName, config['EXPERIMENT']['name'], config['PATH']['output'], config['EXTENSION']['output'])


    print('audio duration: ', audio_duration)
    print('real-time factor: ', tu / audio_duration)


    print(feature_t, SAD_t, KBM_t, clustering_t, reseg_t, tu)

    #wav_path = './audio_test/2.wav'
    #print(config['PATH']['audio']+showName+'.wav')

    
    speakerSlice = getSegResultForPlot('RTTM',frameshift,finalSegmentTable, np.squeeze(finalClusteringTableResegmentation), showName, 'test', 'RTTM', 'rttm')
    
    '''
    p = PlotDiar(map=speakerSlice, wav=wav_path, title = 'Binary key diarization: ' +wav_path   +', number of speakers: ' + str(len(speakerSlice)), gui=True, pick=True, size=(25, 6))
    p.draw()
    p.plot.show()
    ''' 
        
    
    #getSegmentationFile(config['OUTPUT']['format'],config.getfloat('FEATURES','frameshift'),segmentTable, finalClusteringTable[:,bestClusteringID.astype(int)-1], showName, config['EXPERIMENT']['name'], config['PATH']['output'], config['EXTENSION']['output'])      
    
    '''
    if config.getint('OUTPUT','returnAllPartialSolutions'):
        if not os.path.isdir(config['PATH']['output']):
            os.mkdir(config['PATH']['output'])
        outputPathInd = config['PATH']['output']+ config['EXPERIMENT']['name'] + '/' + showName + '/'
        if not os.path.isdir(config['PATH']['output'] + config['EXPERIMENT']['name']):
            os.mkdir(config['PATH']['output'] + config['EXPERIMENT']['name'])
        if not os.path.isdir(outputPathInd):            
            os.mkdir(outputPathInd)
        for i in np.arange(k):
            getSegmentationFile(config['OUTPUT']['format'],config.getfloat('FEATURES','frameshift'), segmentTable, finalClusteringTable[:,i], showName, showName+'_'+str(np.size(np.unique(finalClusteringTable[:,i]),0))+'_spk', outputPathInd, config['EXTENSION']['output'])        
    '''
        
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
    
    if config.getint('GENERAL','performFeatureExtraction'):
        # Audio files are searched at the corresponding folder
        showNameList = sorted(glob.glob(config['PATH']['audio']+'*'+config['EXTENSION']['audio']))
    else:
        # Feature files are searched if feature extraction is disabled:
        showNameList = sorted(glob.glob(config['PATH']['features']+'*'+config['EXTENSION']['features']))
        
    # If the output file already exists from a previous call it is deleted
    if os.path.isfile(config['PATH']['output']+config['EXPERIMENT']['name']+config['EXTENSION']['output']):
        os.remove(config['PATH']['output']+config['EXPERIMENT']['name']+config['EXTENSION']['output'])
                
    # Output folder is created
    if not os.path.isdir(config['PATH']['output']):
        os.mkdir(config['PATH']['output'])

    # Files are diarized one by one
    for idx,showName in enumerate(showNameList):
        if os.path.basename(showName) == '2.wav':
        #if os.path.basename(showName) == 'chinese_same.wav':
            print('\nProcessing file',idx+1,'/',len(showNameList))
            runDiarization(os.path.splitext(os.path.basename(showName))[0], config)
