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

    positioners = [PositionerFactory.create_predictor(config, plotter)]#, PositionerFactory.create_tracker(config, plotter)]

    while not keyboard.is_pressed('esc') and not frame_timer.reached_end():
        delta_time = frame_timer.mark()
        
        for positioner in positioners:
            positioner.update(delta_time)
        
        print(f"Predicted position: {positioners[0].get_position()}")# Tracked position: {positioners[1].get_position()}")

    del positioners

    # if config['offline']:
    #     return

    #plotter.print_metrics()
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
    offline = False
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
        
        one_dim = []
        two_dims = []
        for r in results:
            if 'Mean error Y: ' in r[0]:
                two_dims.append(r)
            else:
                one_dim.append(r)

        one_dim = sorted(one_dim, key=lambda t: (t[1], t[0]['Mean error X: ']))
        two_dims = sorted(two_dims, key=lambda t: (t[1], t[0]['Mean error Y: ']))
        
        print("\n")        

        result_string = "-" * 20 + " 1D " + "-" * 20
        for i, result in enumerate(one_dim):
            metrics, description, options = result
            result_string += "Results for " + description + str(options) + " = " + str(metrics) + "\n"
            # add method of doppler per example in config and select best (min error ) printint that method used
        with open('result.txt', 'w') as f:
            f.write(result_string)

        result_string = "-" * 20 + " 2D " + "-" * 20
        for i, result in enumerate(two_dims):
            metrics, description, options = result
            result_string += "Results for " + description + str(options) + " = " + str(metrics) + "\n"
            # add method of doppler per example in config and select best (min error ) printint that method used
        with open('result.txt', 'a') as f:
            f.write(result_string)
        
        print("\nEnd")

    else:
        plot = False
        plotter = Plotter()
        if plot:
            plotter.load_from_file()
            plotter.plot()
        else:
            config = Config.read_config(offline=False)
            plotter.add_data('config', config)    
            main_loop(plotter, config)