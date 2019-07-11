import matplotlib
import matplotlib.pyplot as plot
from matplotlib.patches import Rectangle
from player import AudioPlayer


class PlotDiar:
    """
    A viewer of segmentation
    """
    def __init__(self, map=None, wav=None, duration=120, gui=False, vgrid=False, size=(18, 9)):
        self.rect_picked = None
        self.rect_color = (0.0, 0.6, 1.0, 1.0)  # '#0099FF'
        self.rect_selected_color = (0.75, 0.75, 0, 1.0)  # 'y'
        self.cluster_colors = [(0.0, 0.6, 1.0, 1.0), (0.0, 1.0, 0.6, 1.0), (0.6, 0.0, 1.0, 1.0), 
                               (0.6, 1.0, 0.0, 1.0), (1.0, 0.0, 0.6, 1.0), (1.0, 0.6, 0.0, 1.0)]

        plot.rcParams['keymap.fullscreen'] = 'ctrl+f'
        plot.rcParams['keymap.home'] = ''
        plot.rcParams['keymap.back'] = ''
        plot.rcParams['keymap.forward'] = ''
        plot.rcParams['keymap.pan'] = ''
        plot.rcParams['keymap.zoom'] = 'ctrl+z'
        plot.rcParams['keymap.quit'] = 'ctrl+q'
        plot.rcParams['keymap.grid'] = ''
        plot.rcParams['keymap.yscale'] = ''
        plot.rcParams['keymap.xscale'] = ''
        plot.rcParams['keymap.all_axes'] = ''
        plot.rcParams['toolbar'] = 'None'
        plot.rcParams['keymap.save'] = 'ctrl+s'
        # plot.rcParams.update({'font.family': 'courrier'})

    
        self.gui = gui
        self.vgrid = vgrid
        self.fig = plot.figure(figsize=size, facecolor='white', tight_layout=True)
        self.plot = plot
        self.title = 'Binary key diarization: ' +wav_path   +', number of speakers: '

        self.ax = self.fig.add_subplot(1, 1, 1)
        cids = list()
        if self.gui:
            cids.append(
                self.fig.canvas.mpl_connect('key_press_event', self._on_keypress))
            #cids.append(
            #    self.fig.canvas.mpl_connect('button_press_event', self._on_click))

        self.height = 5
        self.maxx = 0
        self.maxy = 0
        
        self.maxx = duration
        
        self.end_play = 0
        self.wav = wav
        self.audio = None
        if self.wav is not None and self.gui:
            self.audio = AudioPlayer(wav)
            self.timer = self.fig.canvas.new_timer(interval=100)
            self.timer.add_callback(self._update_result)
            self.timer.start()

        #self.timeline = self.ax.plot([0, 0], [0, 0], color='r')[-1]
        
        self.allmap = map
        self.map = {}
        
        self.last_second = 0
        
        self.time_stamp = list()
        self.time_stamp_idx = 0
    
    def _update_result(self):
        t = self.audio.time()
        int_t = int(t)
        if int_t > self.last_second and int_t in self.allmap.keys():
            print('drawing: ', int_t)
            self.map = self.allmap[int_t]
            self.draw()
            self.fig.canvas.draw()
            self.last_second = int_t
        elif int_t <= self.last_second:
            print('already plot for:', int_t)
        else:
            print('No data to draw for:', int_t)
            
        self._draw_info(t)
    
    
    def _draw_timeline(self, t):
        """
        Draw the timeline a position t
        :param t: in second, a float
        """
        min, max = self.ax.get_ylim()
        self.timeline.set_data([t, t], [min, max])
        self._draw_info(t)

    def _update_timeline(self):
        """
        Update the timeline given the position in the audio player
        """
        if self.audio is not None and self.audio.playing():
            t = self.audio.time()
            min, max = self.ax.get_xlim()
            if t > self.end_play and self.rect_picked is not None:
                self.audio.pause()
                self.end_play = self.maxx
            self._draw_timeline(t)
            if t > max:
                self._dec_right(min, max)
            if t < min:
                self._dec_left(min, max)
            self.fig.canvas.draw()

    def _draw_info(self, t):
        """
        Draw information on segment and timestamp
        :param t: a float
        :return:
        """
        ch = 'time:{:s} ({:.3f} sec {:d} frame)'.format(self._hms(t), t,
                                                        int(t * 100))
        ch2 = '\n\n\n'


        plot.xlabel(ch + '\n' + ch2)

    def draw(self):
        """
        Draw the segmentation
        """
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

        plot.xlim([0, self.maxx])

        plot.ylim([0, y])
        plot.yticks(labels_pos, labels)
        self.maxy = y
        self.end_play = self.maxx
        
        for cluster in self.map:
            self.ax.plot([0, self.maxx], [y, y], linestyle=':',
                         color='#AAAAAA')
            y -= self.height

        plot.title(self.title +  str(i+1))
        
        #if self.gui:
        #    self._draw_info(0)
        
        plot.tight_layout()
        
        #self.time_stamp = list(set(self.time_stamp))
        #self.time_stamp.sort()
        
        '''
        if self.vgrid:
            for x in  self.time_stamp:
                self.ax.plot([x, x], [0, self.maxy], linestyle=':',
                             color='#AAAAAA')
        '''

    def _dec_right(self, min, max):
        """
        Move right
        :param min: a float
        :param max: a float
        """
        dec = (max - min) // 10
        diff = max - min
        if max + dec <= self.maxx:
            # print('** 1 ', min, max, dec)
            plot.xlim(min + dec, max + dec)
        else:
            # print('** 2 ', min, max, dec, diff)
            plot.xlim(self.maxx - diff, self.maxx)

    def _dec_left(self, min, max):
        """
        Move left
        :param min: a float
        :param max: a float
        """
        dec = (max - min) // 10
        diff = max - min
        if min - dec >= 0:
            plot.xlim(min - dec, max - dec)
        else:
            plot.xlim(0, diff)

    def _on_keypress(self, event):
        """
        manage the keypress event
        :param event: a key event
        """
        hmin, hmax = self.ax.get_xlim()
        diff = hmax - hmin
        if event.key == 'ctrl++' or event.key == 'ctrl+=':
            plot.xlim(hmin * 1.5, hmax * 1.5)
        elif event.key == 'ctrl+-':
            plot.xlim(hmin / 1.5, hmax / 1.5)
        elif event.key == 'escape':
            plot.xlim(0, self.maxx)
            plot.ylim(0, self.maxy)
        elif event.key == 'right':
            self._dec_right(hmin, hmax)
        elif event.key == 'left':
            self._dec_left(hmin, hmax)
        elif event.key == 'ctrl+right':
            plot.xlim(self.maxx - diff, self.maxx)
        elif event.key == 'ctrl+left':
            plot.xlim(0, diff)
        elif event.key == 'alt+right':
            self.time_stamp_idx = min(len(self.time_stamp)-1, self.time_stamp_idx + 1)
            if self.audio is not None:
                self.audio.pause()
                self.audio.seek(self.time_stamp[self.time_stamp_idx])
            self._draw_timeline(self.time_stamp[self.time_stamp_idx])
        elif event.key == 'alt+left':
            self.time_stamp_idx = max(0, self.time_stamp_idx - 1)
            if self.audio is not None:
                self.audio.pause()
                self.audio.seek(self.time_stamp[self.time_stamp_idx])
            self._draw_timeline(self.time_stamp[self.time_stamp_idx])
        elif event.key is None and self.audio is not None:
            self.audio.play()
        elif event.key == ' ' and self.audio is not None:
            if(self.audio.playing()):
                self.audio.pause()
            else:
                self.audio.play()

        self.fig.canvas.draw()


    def _on_click(self, event):
        """
        manage the mouse event
        :param event: a mouse event
        """
        if event.xdata is not None and self.rect_picked is None:
            if self.audio is not None:
                self.audio.pause()
                self.audio.seek(event.xdata)
            self._draw_timeline(event.xdata)
            self.fig.canvas.draw()

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
