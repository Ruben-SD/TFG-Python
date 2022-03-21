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

def main_loop(plotter, config):
    frame_timer = FrameTimer(config, plotter)

    positioners = [PositionerFactory.create_predictor(config, plotter), PositionerFactory.create_tracker(config, plotter)]

    while not keyboard.is_pressed('esc') and not frame_timer.reached_end():
        delta_time = frame_timer.mark()
        
        for positioner in positioners:
            positioner.update(delta_time)
        
        print(f"Predicted position: {positioners[0].get_position()} Tracked position: {positioners[1].get_position()}")

    del positioners

    if config['offline']:
        return

    plotter.print_metrics()
    plotter.plot()
    plotter.save_to_file()

def offline_loop(config):
    print("Running", config['description'] + "...")
    sys.stdout = open(os.devnull, 'w')
    plotter = Plotter()
    main_loop(plotter, config)
    sys.stdout = sys.__stdout__
    #plotter.print_metrics()
    #todo save graph
    #plotter.save_to_file('offlinefolder)
    return plotter.compute_metrics(), config['description'], config['options']


if __name__=="__main__":
    offline = True
    if offline:
        configs = Config.get_all_configs()        
        options = {'kalman_filter': None, 'doppler_threshold': { "values": [1.35, 1.5] }, 'noise_variance_weighted_mean': None, 'outlier_removal': { "values": [1.25, 1.5, 1.75, 2, 2.35]}, 'ignore_spikes': None}
        all_configs = []
        print("Generating all configurations and options combinations...")
        for i in range(1, len(options) + 1):
            current_options = list(map(dict, itertools.combinations(options.items(), i)))
            for conf, opts in list(itertools.product(configs, current_options)):
                for key, val in opts.items():
                    if val is not None and "values" in val:
                        for i in range(len(val["values"])):
                            curr_opts = {key: value for key, value in opts.items()}
                            curr_opts[key]["index"] = i
                            current_config = {key: value for key, value in conf.items()}
                            current_config['options'] = curr_opts
                            all_configs.append(current_config)
        
        print("Total combinations:", len(all_configs), "\n")
        print("Starting threads...")
        pool = multiprocessing.Pool(processes=os.cpu_count())
        results = pool.map(offline_loop, all_configs)    
        
        results = sorted(results, key=lambda t: (t[1], t[0]['Mean error Y: ']))
        
        print("\n")        
        for i, result in enumerate(results):
            metrics, description, options = result
            print("Results for", description, options, " = ", metrics)
            # add method of doppler per example in config and select best (min error ) printint that method used
        print("\nEnd")
    else:
        config = Config.read_config(offline=False)
        plotter = Plotter()
        plotter.add_data('config', config)    
        main_loop(plotter, config)



