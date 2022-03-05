import config
import keyboard
from frametimer import FrameTimer
from positioner import PositionerFactory
from config import Config
from plotter import *

config = Config.read_config()

frame_timer = FrameTimer()

predictor = PositionerFactory.create_predictor(config)
tracker = PositionerFactory.create_tracker(config)

while not keyboard.is_pressed('q'):
    delta_time = frame_timer.mark()
    
    predicted_position = predictor.update(delta_time)
    tracked_position = tracker.update(delta_time)
    
    print(f"Predicted position: {predicted_position} Real position: {tracked_position}")
    
del predictor
del tracker

#plotter.print_metrics()
plotter.plot()
plotter.save_to_file()