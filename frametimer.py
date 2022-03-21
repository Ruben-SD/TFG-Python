import time

class FrameTimer:
    def __init__(self, config, plotter):
        self.offline = config['offline'] 
        self.plotter = plotter
        if not self.offline:
            self.last_frame_time = time.time()
            plotter.add_data('time', [])
            self.start_time = time.time()
        else:
            self.cur_frame = -1
            self.saved_time = config['time']

    def mark(self):
        if not self.offline:
            current_time = time.time()
            delta_time = current_time - self.last_frame_time
            self.plotter.add_sample('time', current_time - self.start_time)
            #print("FPS: ", 1/delta_time)
            self.last_frame_time = time.time()
            return delta_time
        else:
            self.cur_frame += 1
            if self.cur_frame == 0:
                self.plotter.add_sample('time', self.saved_time[0])
                return self.saved_time[0]
            else:
                self.plotter.add_sample('time', self.saved_time[self.cur_frame])
                return self.saved_time[self.cur_frame] - self.saved_time[self.cur_frame - 1]

    def reached_end(self):
        if not self.offline:
            return False
        else:
            return self.cur_frame == len(self.saved_time) - 1
        