import sys
import configparser
import threading
import time
import librosa
import numpy as np
from diarizationFunctions import *


class ThreadingKBM(threading.Thread):

    def __init__(self, config):
        super(ThreadingKBM,self).__init__()
        
        self.daemon = True
        
        #thread = threading.Thread(target=self.run, args=())
        #thread.daemon = True                            # Daemonize thread
        #thread.start()                                  # Start the execution
        
        print('initiating the KBM thread')
        
        self.KBM_window_length = config.getint('KBM','windowLength')
        self.KBM_minG = config.getint('KBM','minimumNumberOfInitialGaussians')
        self.KBM_minW = config.getint('KBM','maximumKBMWindowRate')
        self.KBM_rel_size = config.getfloat('KBM','relKBMsize')
        self.KBM_topG = config.getint('BINARY_KEY','topGaussiansPerFrame')
        
        self.data = []
        self.nSpeechFeatures = 0
        self.last_second = 0
        self.this_second = 0
        
        
        self.kbmSize = None
        self.kbm = None
        self.gmPool = None
        self.Vg = None
        self.kbm_version = 0
        
    def run(self):
        while True:
            #print('Fail:this and last second: ',  self.this_second, self.last_second, ' data: ',len(self.data))
            
            if self.this_second > self.last_second and self.this_second > 11:
        
                t0 = time.time()
                
                if np.floor((self.nSpeechFeatures-self.KBM_window_length)/self.KBM_minG) < self.KBM_minW:
                    windowRate = int(np.floor((np.size(self.data,0)-self.KBM_window_length)/self.KBM_minG))
                else:
                    windowRate = self.KBM_minW     
          
                poolSize = np.floor((self.nSpeechFeatures-self.KBM_window_length)/windowRate)
                
                self.kbmSize = 320
                #self.kbmSize = int(np.floor(poolSize*self.KBM_rel_size))
                self.kbm, self.gmPool = trainKBM(self.data,self.KBM_window_length, windowRate,self.kbmSize ) 
 
                
                self.Vg = getVgMatrix(self.data,self.gmPool,self.kbm,self.KBM_topG)
                
                # update the KBM version
                self.kbm_version = self.this_second
                self.last_second = self.this_second
            
            else:
                time.sleep(0.5)

def main(filename, config, Plot_and_play = False):
    

    
    wav_path = config['PATH']['audio']+filename
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
    

    KBM_topG = config.getint('BINARY_KEY','topGaussiansPerFrame')
    bk_bits = config.getfloat('BINARY_KEY','bitsPerSegmentFactor')
    init_cluster = config.getint('CLUSTERING','N_init')
    metric = config['CLUSTERING']['metric']
    
   
    step = 1
    i = step
    y = [] 
    
    
    kbm = None
    gmPool = None
    Vg = None
    kbm_version = 0

    # init the threading
    
    kbm_t = ThreadingKBM(config)
    kbm_t.start()
    
 
    
    # init the viewer and player
    if Plot_and_play:
        p = PlotDiar(map={}, wav=wav_path, duration = audio_duration+5, title = 'Binary key diarization: ' +wav_path   +', number of speakers: ', gui=True)
        wm = p.plot.get_current_fig_manager()
        wm.window.state('zoomed')
        p.plot.show()
        p.audio.play()
    
    whole_process_start_time = time.time()
    
    while i < audio_duration:
        
        
        
        start_time = time.time()
        
        if i + step >= audio_duration:
            y = y_total
        else:
            y = y_total[0 : i * sr]
            
        i = i + step
        
        
        allData = extractFeaturesFromSignal(y, sr, nfilters,ncoeff, NFFT, hop)
        nFeatures = allData.shape[0]    
                
        maskUEM = np.ones([1,nFeatures])     
        maskSAD = getSADfromSignal(y, sr, frameshift, nFeatures)
        
        mask = np.logical_and(maskUEM, maskSAD)    
        mask = mask[0][0:nFeatures]
        
        nSpeechFeatures=np.sum(mask)
        speechMapping = np.zeros(nFeatures)

        speechMapping[np.nonzero(mask)] = np.arange(1,nSpeechFeatures+1)
        data=allData[np.where(mask==1)]
        del allData        
        
        segmentTable=getSegmentTable(mask,speechMapping, seg_length, seg_incre, seg_rate)
        numberOfSegments=np.size(segmentTable,0)
        

        start_time = time.time()
        
        # update data in the thread of kbm training
        kbm_t.nSpeechFeatures = nSpeechFeatures
        kbm_t.data = data
        kbm_t.this_second = i
        
        
        
        
        
        if kbm_t.Vg is not None:
            print('i: ', i, ' kbm second: ', kbm_t.last_second, ' kbm: ', kbm_t.kbmSize)
            
            if kbm_t.kbm_version > kbm_version:
                print('update kbm now:', kbm_version,'-->',kbm_t.kbm_version)
                # update kbm, gmPool and Vg
                kbm = kbm_t.kbm
                gmPool = kbm_t.gmPool
                kbmSize = kbm_t.kbmSize
                Vg = kbm_t.Vg
                kbm_version = kbm_t.kbm_version
            
            Vg_len = np.size(Vg, 0)            
            data_len = np.size(data, 0)
            if data_len > Vg_len:
                # get Vg for new input data now
                new_Vg = getVgMatrix(data[Vg_len:, :],gmPool,kbm, KBM_topG)
                # combine new_Vg with Vg
                
                prev_vg_shape = Vg.shape
                
                Vg = np.vstack((Vg, new_Vg))

                print('data:', data_len,  new_Vg.shape,  '+', prev_vg_shape ,'-->', Vg.shape, ' kbm version:', kbm_version)

            
            segmentBKTable, segmentCVTable = getSegmentBKs(segmentTable, kbmSize, Vg, bk_bits, speechMapping)    
            
            
            
            initialClustering = np.digitize(np.arange(numberOfSegments),np.arange(0,numberOfSegments,numberOfSegments/init_cluster))
            finalClusteringTable, k = performClustering(speechMapping, segmentTable, segmentBKTable, segmentCVTable, Vg, bk_bits, kbmSize, init_cluster, initialClustering, metric)        
            bestClusteringID = getBestClustering(metric, segmentBKTable, segmentCVTable, finalClusteringTable, k)
            
            finalClustering = finalClusteringTable[:,bestClusteringID.astype(int)-1]
            
            if config.getint('RESEGMENTATION','resegmentation') and np.size(np.unique(finalClustering),0)>1:

            
                finalClusteringTableResegmentation,finalSegment = performResegmentation(data,speechMapping, mask,finalClustering,segmentTable,config.getint('RESEGMENTATION','modelSize'),config.getint('RESEGMENTATION','nbIter'),config.getint('RESEGMENTATION','smoothWin'),nSpeechFeatures)
                finalClustering = np.squeeze(finalClusteringTableResegmentation)
            
            else:
                finalClustering = rearrangeClusterID(finalClustering)
                finalSegment = segmentTable
            
            
            if Plot_and_play:
                speakerSlice = getSegResultForPlot(frameshift,finalSegment, finalClustering)
                p.allmap[i] = speakerSlice
            
            else:
                tu = time.time() - start_time
                getSegResultForPlotlater(frameshift,finalSegment, finalClustering, filename, i, tu)
     
            
        used_time = time.time() - start_time
        #time.sleep(0.5)
        
        if used_time > 1:
            pass
            print('[Failed] second: ', i, ' used time:', used_time)
        else:
            print('second: ', i, ' used time:', used_time)
            time.sleep(1 - used_time)
    
    
    print('total:', time.time() - whole_process_start_time)

if __name__ == "__main__":     
    # If a config file in INI format is passed by argument line then it's used. 
    # For INI config formatting please refer to https://docs.python.org/3/library/configparser.html
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]
    else:
        configFile = 'config-test.ini'    
    config = configparser.ConfigParser()
    config.read(configFile)
    
    
    filename = '3057402.wav'
    main(filename, config, Plot_and_play = False)
