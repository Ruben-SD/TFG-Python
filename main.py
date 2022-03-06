import config
import keyboard
from frametimer import FrameTimer
from positionerfactory import PositionerFactory
from config import Config
from plotter import *
from matplotlib import pyplot as plt
import pickle
import scipy.fft

with open('sound', 'rb') as f:
   sound = np.array(pickle.load(f))
print(int(len(sound)/2))

def some_function(l, n):
    l.extend([0] * n)
    l = l[:n]
    return l

delayed = some_function(list(sound[int(len(sound)/2):]), len(sound))

af = scipy.fft.fft(sound)
bf = scipy.fft.fft(delayed)
c = scipy.fft.ifft(af * scipy.conj(bf))

time_shift = np.argmax(abs(c))
print("TIMESHIFT",time_shift)

plt.plot(np.arange(len(sound)), sound) 
plt.show()




config = Config.read_config()

frame_timer = FrameTimer()

positioners = [PositionerFactory.create_predictor(config), PositionerFactory.create_tracker(config)]

while not keyboard.is_pressed('q'):
    delta_time = frame_timer.mark()
    
    for positioner in positioners:
        positioner.update(delta_time)
    
    print(f"Predicted position: {positioners[0].get_position()} Tracked position: {positioners[1].get_position()}")
    
del positioners

plotter.print_metrics()
plotter.plot()
plotter.save_to_file()