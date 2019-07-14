import matplotlib
import matplotlib.pyplot as plot
from matplotlib.patches import Rectangle
from player import AudioPlayer


class PlotDiar:
    """
    A viewer of segmentation
    """
    def __init__(self, thread = None, wav=None, duration=120, gui=False, vgrid=False, size=(18, 9)):
        self.rect_picked = None
        self.rect_color = (0.0, 0.6, 1.0, 1.0)  # '#0099FF'
        self.rect_selected_color = (0.75, 0.75, 0, 1.0)  # 'y'
        self.cluster_colors = [(0.0, 0.6, 1.0, 1.0), (0.0, 1.0, 0.6, 1.0), (0.6, 0.0, 1.0, 1.0), 
                               (0.6, 1.0, 0.0, 1.0), (1.0, 0.0, 0.6, 1.0), (1.0, 0.6, 0.0, 1.0)]

    
        self.gui = gui
        self.vgrid = vgrid
        self.fig = plot.figure(figsize=size, facecolor='white', tight_layout=True)
        self.plot = plot

        self.ax = self.fig.add_subplot(1, 1, 1)

        self.height = 5
        self.maxx = 0
        self.maxy = self.height * 5
        
        self.maxx = duration
        
        self.ax.set_xlim(0, self.maxx)
        self.ax.set_ylim(0, self.maxy)
        
        self.wav = wav
        self.audio = None

        
        #self.plot.ion()
        #self.plot.show()
        #self.timeline = self.ax.plot([0, 0], [0, 0], color='r')[-1]
        
        self.title = 'Binary key diarization: ' +self.wav   +', number of speakers: '

        self.diarization_thread = thread
        self.allmap = thread.allmap
        self.map = {}
        
        self.last_second = 0
        
        self.time_stamp = list()
        self.time_stamp_idx = 0
        
        if self.wav is not None:
            self.audio = AudioPlayer(wav)
            self.timer = self.fig.canvas.new_timer(interval=100)
            self.timer.add_callback(self._update_result)
            self.timer.start()
        
        
        #self.plot.ion()
        #self.plot.show()
        #.plot.pause(0.1)
    
    def _update_result(self):
        self.allmap = self.diarization_thread.allmap
        t = self.audio.time()
        int_t = int(t)
        if int_t > self.last_second and int_t in self.allmap.keys():
            print('drawing: ', int_t, 'thread_t:', self.diarization_thread.i)
            self.map = self.allmap[int_t]
            self.draw()
            #self.fig.canvas.draw()
            self.last_second = int_t
        '''
        elif int_t <= self.last_second:
            print('already plot for:', int_t)
        else:
            
            print('No data to draw for:', int_t)
        '''
            
        self._draw_info(t)
        '''
        self.plot.ion()
        self.plot.show()
        self.plot.pause(0.01)
        '''
        
    def _draw_info(self, t):
        """
        Draw information on segment and timestamp
        :param t: a float
        :return:
        """
        ch = 'time:{:s}'.format(self._hms(t))
                                                     
        ch2 = '\n\n\n'


        plot.xlabel(ch + '\n' + ch2)
        self.fig.canvas.draw()


    def draw(self):
        """
        Draw the segmentation
        """
        self.ax.clear()
        self.ax.set_xlim(0, self.maxx)
        
        y = 0
        labels_pos = []
        labels = []
        for i,cluster in enumerate(sorted(self.map.keys())):
            labels.append(cluster)
            labels_pos.append(y + self.height // 2)
            for row in self.map[cluster]:
                x = row['start'] /1000
                #self.time_stamp.append(x)
                #self.time_stamp.append(row['stop'] /1000)
                w = row['stop'] /1000 - row['start'] /1000
                
                #self.maxx = max(self.maxx, row['stop'] /1000)
                
                c = self.cluster_colors[i%len(self.cluster_colors)]
                rect = plot.Rectangle((x, y), w, self.height,
                                      color=c)
                self.ax.add_patch(rect)
            y += self.height

        #plot.xlim([0, self.maxx])

        plot.ylim([0, max(y, self.maxy)])
        plot.yticks(labels_pos, labels)
        #self.maxy = y
        
        for cluster in self.map:
            self.ax.plot([0, self.maxx], [y, y], linestyle=':',
                         color='#AAAAAA')
            y -= self.height

        plot.title(self.title +  str(i+1))
                
        plot.tight_layout()
        
        self._draw_info(t = self.audio.time())

        
        #self.time_stamp = list(set(self.time_stamp))
        #self.time_stamp.sort()
        
        '''
        if self.vgrid:
            for x in  self.time_stamp:
                self.ax.plot([x, x], [0, self.maxy], linestyle=':',
                             color='#AAAAAA')
        '''


    


    

    @classmethod
    def _colors_are_equal(cls, c1, c2):
        """
        Compare two colors
        """
        for i in range(4):
            if c1[i] != c2[i]:
                return False
        return True

    @classmethod
    def _hms(cls, s):
        """
        conversion of seconds into hours, minutes and secondes
        :param s:
        :return: int, int, float
        """
        h = int(s) // 3600
        s %= 3600
        m = int(s) // 60
        s %= 60
        return '{:d}:{:d}:{:.2f}'.format(h, m, s)
