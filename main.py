import sys
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import config
import keyboard
from frametimer import FrameTimer
from positionerfactory import PositionerFactory
from config import Config
import multiprocessing
from plotting import Plotter

def main_loop(config):
    plotter = Plotter()

    frame_timer = FrameTimer(config, plotter)

    positioners = [PositionerFactory.create_predictor(config, plotter), PositionerFactory.create_tracker(config, plotter)]

    while not keyboard.is_pressed('esc') and not frame_timer.reached_end():
        delta_time = frame_timer.mark()
        
        for positioner in positioners:
            positioner.update(delta_time)
        
        print(f"Predicted position: {positioners[0].get_position()} Tracked position: {positioners[1].get_position()}")

    del positioners

    if config['offline']:
        return plotter

    plotter.print_metrics()
    plotter.plot()
    plotter.save_to_file()

def offline_loop(config):
    print("\nRunning", config['description'], "...")
    sys.stdout = open(os.devnull, 'w')
    plotter = main_loop(config)
    sys.stdout = sys.__stdout__
    print("\nResults for", config['description'], ":\n")
    plotter.print_metrics()


if __name__=="__main__":
    offline = True
    if offline:
        configs = Config.get_all_configs()        
        pool = multiprocessing.Pool(processes=len(configs))
        result = pool.map(offline_loop, configs)         
        print("\nEnd")
    else:
        config = Config.read_config(offline=True)
        main_loop(config)



