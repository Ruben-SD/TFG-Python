import time
import plotter
from plotter import *

class FrameTimer:
    def __init__(self):
        self.last_frame_time = time.time()
        plotter.add_data('time', [])
        self.start_time = time.time()

    def mark(self):
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        plotter.add_sample('time', current_time - self.start_time)
        #print("FPS: ", 1/delta_time)
        self.last_frame_time = time.time()
        return delta_time