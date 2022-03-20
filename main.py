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
import itertools

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
    print("Running", config['description'] + "...")
    sys.stdout = open(os.devnull, 'w')
    plotter = main_loop(config)
    sys.stdout = sys.__stdout__
    #plotter.print_metrics()
    return plotter.compute_metrics(), config['description'], config['options']


if __name__=="__main__":
    offline = True
    if offline:
        configs = Config.get_all_configs()        
        options = ['doppler_threshold', 'noise_variance_weighted_mean', 'outlier_removal']
        all_configs = []
        print("Generating all configurations and options combinations...")
        for i in range(1, len(options) + 1):
            current_options = list(itertools.combinations(options, i))
            for conf, opt in list(itertools.product(configs, current_options)):
                import copy
                current_config = copy.deepcopy(conf)
                current_config['options'] = opt
                all_configs.append(current_config)
        
        print("Total combinations:", len(all_configs), "\n")
        print("Starting threads...")
        pool = multiprocessing.Pool(processes=len(all_configs))
        results = pool.map(offline_loop, all_configs)    
        
        results = sorted(results, key=lambda t: (t[1], t[0]['Mean error X: ']))
        
        print("\n")        
        for i, result in enumerate(results):
            metrics, description, options = result
            print("Results for", description, options, " = ", metrics)
            # add method of doppler per example in config and select best (min error ) printint that method used
        print("\nEnd")
    else:
        config = Config.read_config(offline=True)
        main_loop(config)



