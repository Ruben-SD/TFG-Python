import config
import keyboard
from frametimer import FrameTimer
from predictor import Predictor
from camerasystem import CameraSystem
from config import config
from plotter import *

frame_timer = FrameTimer()

predictor = Predictor(config)
ground_truth = CameraSystem(config)

while not keyboard.is_pressed('q'):
    delta_time = frame_timer.mark()
    
    predicted_position = predictor.update_position(delta_time)
    #real_position = ground_truth.update_position(delta_time)
    
    print(f"Predicted position: {predicted_position}")# Real position: {real_position}")
    
del predictor
del ground_truth

plotter.print_metrics()
plotter.plot()
plotter.save_to_file()