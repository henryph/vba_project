import sys
import configparser
import threading
import time
import librosa
import numpy as np
from diarizationFunctions import *
sys.path.append('visualization')
from viewer3 import PlotDiar

class ThreadingPlot(threading.Thread):
    '''
    This is the thread to Plot the result
    '''
    def __init__(self, wav_path, audio_duration):
        super(ThreadingPlot,self).__init__()
        self.daemon = True
        
        print('initiating the Plotting thread')
        
        self.p = PlotDiar(map={}, wav=wav_path, duration = audio_duration+5, gui=True)
        #wm = self.p.plot.get_current_fig_manager()
        #wm.window.state('zoomed')
        #self.p.fig.show()
        self.p.plot.ion()
        self.p.plot.show()
        self.p.audio.play()
        
        
    def run(self):
        while True:
            self.p.update_result()
            #self.p.plot.show()
            self.p.plot.pause(0.01)
            time.sleep(0.2)


class ThreadingClustering(threading.Thread):
    '''
    This is the thread to run AHC clustering
    '''
    def __init__(self, config):
        super(ThreadingClustering,self).__init__()
        self.daemon = True
        
        print('initiating the Clustering thread')
        
        self.init_cluster = config.getint('CLUSTERING','N_init')
        self.metric = config['CLUSTERING']['metric']
        self.bk_bits = config.getfloat('BINARY_KEY','bitsPerSegmentFactor')
        self.reseg = config.getint('RESEGMENTATION','resegmentation')
        
        self.last_second = 0
        self.this_second = 0
        
        self.finalClustering = None
        self.finalSegment = None
        self.prev_clustering = None
        
    def run(self):
        while True:
            if self.this_second > self.last_second:
                
                segmentTable = self.segmentTable.copy()
                numberOfSegments = np.size(segmentTable,0)

                initialClustering = np.digitize(np.arange(numberOfSegments),np.arange(0,numberOfSegments,numberOfSegments/self.init_cluster))
                finalClusteringTable, k = performClustering(self.speechMapping, segmentTable, self.segmentBKTable, self.segmentCVTable, self.Vg, self.bk_bits, self.kbmSize, self.init_cluster, initialClustering, self.metric)        
                bestClusteringID = getBestClustering(self.metric, self.segmentBKTable, self.segmentCVTable, finalClusteringTable, k)

                finalClustering = finalClusteringTable[:,bestClusteringID.astype(int)-1]

                if self.reseg and np.size(np.unique(finalClustering),0)>1:
                    '''
                    Fix later
                    '''
                    finalClusteringTableResegmentation,self.finalSegment = performResegmentation(data,speechMapping, mask,finalClustering,segmentTable,config.getint('RESEGMENTATION','modelSize'),config.getint('RESEGMENTATION','nbIter'),config.getint('RESEGMENTATION','smoothWin'),nSpeechFeatures)
                    self.finalClustering = np.squeeze(finalClusteringTableResegmentation)

                else:
                    
                    
                    if self.prev_clustering is not None:
                        self.finalClustering = reselectBestClustering(finalClusteringTable, bestClusteringID, self.prev_clustering, self.segmentCVTable, mode = 1)
                    
                    #print('seg', numberOfSegments, 'cluster:', finalClustering.shape)
                    
                    else:
                        self.finalClustering = rearrangeClusterID(finalClustering)
                    
                    self.prev_clustering = self.finalClustering.copy()
                    self.finalSegment = segmentTable.copy()
                    
                    
                    
                    
                # update last second to indicate the clustering completes
                self.last_second = self.this_second

            else:
                time.sleep(0.1)
        
    
class ThreadingKBM(threading.Thread):
    '''
    This is the thread to run KBM training and calculate CV for all feature vectors avialable
    '''
    def __init__(self, config):
        super(ThreadingKBM,self).__init__()
        
        self.daemon = True
        
        #thread = threading.Thread(target=self.run, args=())
        #thread.daemon = True                            # Daemonize thread
        #thread.start()                                  # Start the execution
        
        print('initiating the KBM thread')
        
        self.KBM_window_length = config.getint('KBM','windowLength')
        self.KBM_minG = config.getint('KBM','minimumNumberOfInitialGaussians')
        self.KBM_maxW = config.getint('KBM','maximumKBMWindowRate')
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
            
            if self.this_second > self.last_second:
                
                windowRate = int(np.floor((np.size(self.data,0)-self.KBM_window_length)/self.KBM_minG))
                
                if windowRate > 0:
                        
                
                    windowRate = min(self.KBM_maxW, windowRate)
                    
                    '''
                    if np.floor((self.nSpeechFeatures-self.KBM_window_length)/self.KBM_minG) < self.KBM_maxW:
                        windowRate = int(np.floor((np.size(self.data,0)-self.KBM_window_length)/self.KBM_minG))
                    else:
                        windowRate = self.KBM_maxW     
                    '''
                
                    #poolSize = np.floor((self.nSpeechFeatures-self.KBM_window_length)/windowRate)

                    self.kbmSize = 320
                    #self.kbmSize = int(np.floor(poolSize*self.KBM_rel_size))
                    self.kbm, self.gmPool = trainKBM(self.data,self.KBM_window_length, windowRate,self.kbmSize ) 


                    self.Vg = getVgMatrix(self.data,self.gmPool,self.kbm,self.KBM_topG)

                    # update the KBM version
                    self.kbm_version = self.this_second
                    self.last_second = self.this_second

                else:
                    time.sleep(0.5)
            else:
                time.sleep(0.5)

class mainThread(threading.Thread):
    '''
    This is the thread to run KBM training and calculate CV for all feature vectors avialable
    '''
    def __init__(self, filename, config, Use_clustering_thread=False, Text_output = False):
        super(mainThread,self).__init__()
        
        self.daemon = True
        
        self.allmap = {}
        wav_path = config['PATH']['audio']+filename
        self.y_total, self.sr = librosa.load(wav_path,sr=None)
        self.audio_duration = librosa.get_duration(self.y_total, sr=self.sr)
        
        
        
        
        framelength = config.getfloat('FEATURES','framelength')
        self.frameshift = config.getfloat('FEATURES','frameshift')
        self.nfilters = config.getint('FEATURES','nfilters')
        self.ncoeff = config.getint('FEATURES','ncoeff')
        
        self.nFeaturesPerSecond = int(self.frameshift * 10000)
        
        frame_length_inSample=framelength*self.sr
        self.hop = int(self.frameshift*self.sr)
        self.NFFT=int(2**np.ceil(np.log2(frame_length_inSample)))
        
        self.seg_length = config.getint('SEGMENT','length')
        self.seg_incre = config.getint('SEGMENT','increment')
        self.seg_rate = config.getint('SEGMENT','rate')
        
        
        self.KBM_topG = config.getint('BINARY_KEY','topGaussiansPerFrame')
        self.bk_bits = config.getfloat('BINARY_KEY','bitsPerSegmentFactor')
        self.init_cluster = config.getint('CLUSTERING','N_init')
        self.metric = config['CLUSTERING']['metric']
        
        
        
        self.kbm = None
        self.gmPool = None
        self.Vg = None
        self.kbm_version = 0
        
        
        # init the threading
    
        self.kbm_t = ThreadingKBM(config)
        self.kbm_t.start()
        
        self.Use_clustering_thread = Use_clustering_thread
        self.Text_output = Text_output
        
        if self.Use_clustering_thread:
            self.cluster_t = ThreadingClustering(config)
            self.cluster_t.start()
        
        self.features = np.empty((0, self.ncoeff))
        self.maskSAD = np.empty((1,0))
        
        self.prev_clustering = None
        
        self.i = 1
        self.step = 1
        
    def run(self):
            while self.i < self.audio_duration:
                start_time = time.time()
                
                
                if self.i + self.step >= self.audio_duration:
                    #y = self.y_total
                    new_y = self.y_total[(self.i-self.step) * self.sr:]

                else:
                    #y = self.y_total[0 : self.i * self.sr - 1]
                    new_y = self.y_total[(self.i-self.step) * self.sr : self.i*self.sr - 1]
                
                #print(self.i, len(y)/self.sr, len(new_y), len(y), len(self.y_total))
                self.i = self.i + self.step
                
                f = extractFeaturesFromSignal(new_y, self.sr, self.nfilters,self.ncoeff, self.NFFT, self.hop)
                self.features = np.concatenate((self.features,f), axis = 0)
                
                
                
                
                #allData = extractFeaturesFromSignal(y, self.sr, self.nfilters,self.ncoeff, self.NFFT, self.hop)
                
                allData = self.features
                #print((self.features == allData).all(),self.features.shape, allData.shape)
                #print( self.features.shape, allData.shape)
                
                nFeatures = allData.shape[0]    
                      
                #print(i, len(y), 'Feature:',time.time()-start_time)      
                  
                #maskUEM = np.ones([1,nFeatures]) 
                    
                #maskSAD = getSADfromSignal(y, self.sr, self.frameshift, nFeatures)
                new_maskSAD = getSADfromSignal(new_y, self.sr, self.frameshift, self.nFeaturesPerSecond)
                self.maskSAD = np.concatenate((self.maskSAD, new_maskSAD), axis = 1)
                
                #print('SAD', maskSAD.shape, new_maskSAD.shape, self.maskSAD.shape, (self.maskSAD==maskSAD).all())
                #print('SAD', maskSAD.shape, new_maskSAD.shape, self.maskSAD.shape)

                
                mask = self.maskSAD
                #mask = np.logical_and(maskUEM, maskSAD)    
                mask = mask[0][0:nFeatures]
                
                nSpeechFeatures=np.sum(mask)
                speechMapping = np.zeros(nFeatures)
        
                speechMapping[np.nonzero(mask)] = np.arange(1,nSpeechFeatures+1)
                data=allData[np.where(mask==1)]
                del allData        
                
                segmentTable=getSegmentTable(mask,speechMapping, self.seg_length, self.seg_incre, self.seg_rate)
                numberOfSegments=np.size(segmentTable,0)
                
                PreProcess_t = time.time() - start_time
                
                
                # update data in the thread of kbm training
                self.kbm_t.nSpeechFeatures = nSpeechFeatures
                self.kbm_t.data = data
                self.kbm_t.this_second = self.i
                
                
                if self.kbm_t.Vg is not None:
                    #print('i: ', self.i, ' kbm second: ', self.kbm_t.last_second, ' kbm: ', self.kbm_t.kbmSize)
                    
                    t0 = time.time()
                    if self.kbm_t.kbm_version > self.kbm_version:
                        #print('update kbm now:', self.kbm_version,'-->',self.kbm_t.kbm_version)
                        # update kbm, gmPool and Vg
                        self.kbm = self.kbm_t.kbm
                        self.gmPool = self.kbm_t.gmPool
                        kbmSize = self.kbm_t.kbmSize
                        self.Vg = self.kbm_t.Vg
                        self.kbm_version = self.kbm_t.kbm_version
                    
                    Vg_len = np.size(self.Vg, 0)            
                    data_len = np.size(data, 0)
                    if data_len > Vg_len:
                        
                        # get Vg for new input data now
                        t = time.time()
                        new_Vg = getVgMatrix(data[Vg_len:, :],self.gmPool,self.kbm, self.KBM_topG)
                        vg_time2 = time.time() - t
                        
                        # combine new_Vg with Vg
                        self.Vg = np.vstack((self.Vg, new_Vg))
        
                        #print('data:', data_len,  new_Vg.shape,  '+', Vg_len ,'-->', self.Vg.shape, ' kbm version:',self.kbm_version)
        
                    t1 = time.time()
                    vg_time = t1 - t0
                    
                    segmentBKTable, segmentCVTable = getSegmentBKs(segmentTable, kbmSize, self.Vg, self.bk_bits, speechMapping)    
                    
                    t2 = time.time()
                    seg_time = t2 - t1
                    
                    if self.Use_clustering_thread:
                        # update data in the thread of clustering
                        
                        self.cluster_t.speechMapping = speechMapping
                        self.cluster_t.segmentTable = segmentTable
                        self.cluster_t.segmentBKTable = segmentBKTable
                        self.cluster_t.segmentCVTable = segmentCVTable
                        self.cluster_t.Vg = self.Vg
                        self.cluster_t.kbmSize = kbmSize
                        self.cluster_t.this_second = self.i
                        
                        while time.time() - start_time < 0.90:
                            if self.cluster_t.last_second == self.i:
                                # offline clustering of second i is completed
                                break
                            time.sleep(0.01)
                        else:
                            print('no time to wait')
                        
                        finalClustering = self.cluster_t.finalClustering
                        finalSegment = self.cluster_t.finalSegment
                        #print('Checking:', ' seg:', finalSegment.shape, 'C:', len(finalClustering))
                        if self.cluster_t.last_second < self.i:
                            diff_t = self.i - self.cluster_t.last_second
                            # extend time of the last segment
                            
                            
                            #print('AHC up to ', self.cluster_t.last_second, '+', diff_t, 'seg', segmentTable.shape, 'final seg:', finalSegment.shape, 'C:', len(finalClustering))
                        else:
                            pass
                            #print('Success within 1s; AHC:' ,self.cluster_t.last_second, self.i, 'seg', segmentTable.shape,' final seg:', finalSegment.shape, 'C:', len(finalClustering))
                    
                    else:
                        
                    
                        initialClustering = np.digitize(np.arange(numberOfSegments),np.arange(0,numberOfSegments,numberOfSegments/self.init_cluster))
                        finalClusteringTable, k = performClustering(speechMapping, segmentTable, segmentBKTable, segmentCVTable, self.Vg, self.bk_bits, kbmSize, self.init_cluster, initialClustering, self.metric)        
                        bestClusteringID = getBestClustering(self.metric, segmentBKTable, segmentCVTable, finalClusteringTable, k)
        
                        finalClustering = finalClusteringTable[:,bestClusteringID.astype(int)-1]
        
                        if config.getint('RESEGMENTATION','resegmentation') and np.size(np.unique(finalClustering),0)>1:
        
        
                            finalClusteringTableResegmentation,finalSegment = performResegmentation(data,speechMapping, mask,finalClustering,segmentTable,config.getint('RESEGMENTATION','modelSize'),config.getint('RESEGMENTATION','nbIter'),config.getint('RESEGMENTATION','smoothWin'),nSpeechFeatures)
                            finalClustering = np.squeeze(finalClusteringTableResegmentation)
        
                        else:
                            finalClustering = rearrangeClusterID(finalClustering)
                            finalSegment = segmentTable
        
                    
                    t3 = time.time()
                    c_time = t3 - t2
                                        
                    if finalClustering is not None:
                        speakerSlice = getSegResultForPlot(self.frameshift,finalSegment, finalClustering)
                        self.allmap[self.i] = speakerSlice
                        
                        #self.prev_clustering = finalClustering
                        
                    if self.Text_output:
                        tu = time.time() - start_time
                        getSegResultForPlotlater(self.frameshift,finalSegment, finalClustering, filename, self.i, tu)
                    
                    t4 = time.time()
                    output_time = t4-t3
                    
                used_time = time.time() - start_time
                #time.sleep(0.5)
                
                if used_time > 1:
                    print('[Failed] second: ', self.i, ' used time:', used_time, PreProcess_t, vg_time, vg_time2, seg_time, c_time, output_time, '-' * 20)
                    
                else:
              
                    print('second: ', self.i, ' used time:', used_time, '-' * 20)
                    time.sleep(max(0.98 - used_time, 0))
            
def main(filename, config, Use_clustering_thread, Text_output):
    
    t = mainThread(filename, config, Use_clustering_thread, Text_output)
    t.start()
    
    wav_path = config['PATH']['audio']+filename
    
    y, sr = librosa.load(wav_path,sr=None)
    duration = librosa.get_duration(y, sr=sr)
    
    #print('sr:', sr, 'duration', duration)
    
    p = PlotDiar(thread = t, wav=wav_path, duration = duration+5)
    wm = p.plot.get_current_fig_manager()
    wm.window.state('zoomed')
    p.audio.play()
    p.plot.show()
    
    #time.sleep(180)

if __name__ == "__main__":     
    # If a config file in INI format is passed by argument line then it's used. 
    # For INI config formatting please refer to https://docs.python.org/3/library/configparser.html
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]
    else:
        configFile = 'config-test.ini'    
    config = configparser.ConfigParser()
    config.read(configFile)
    
    
    filename = '3065554.wav'
    filename = '3066806.wav' # poor later
    
    main(filename, config, Use_clustering_thread = True, Text_output = False)
    
