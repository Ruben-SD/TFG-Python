import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
from config import Config
import multiprocessing
import itertools
import ujson
from main import main_loop
plt.rcParams["figure.figsize"] = [16, 9]
plt.rcParams["figure.dpi"] = 100
#pos
# plt.rcParams.update({'font.size': 25,
#                      'legend.fontsize': 22,
#                      'legend.handlelength': 1.75})


#speed
# plt.rcParams.update({'font.size': 35,
#                      'legend.fontsize': 32,
#                      'legend.handlelength': 1.75})                     

# alls peeds
plt.rcParams.update({'font.size': 35,
                     'legend.fontsize': 23,
                     'legend.handlelength': 1.75})       

class Plotter:
    def __init__(self) -> None:
        self.start_timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
        self.data_dictionary = {}
        self.SAVED_DATA_PATH = sys.path[0] + '/saved_data/'
        self.data_dictionary['data_names_to_plot'] = []
        self.metrics = None

    def plot(self):
        self.generate_figure()
        plt.show()

    def plot_position(self):
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Posición (cm)")
        plt.title("Posición vs. tiempo")
        plt.grid()
        time_data = self.data_dictionary['time']
        for data_name in self.data_dictionary['data_names_to_plot']:
            if 'position' in data_name or 'osición' in data_name:
                new_data_name = data_name
                if 'Predictor_position' in data_name:
                    new_data_name = 'Predicción posición ' + data_name[-1]
                elif 'Tracker_position' in data_name:
                    new_data_name = 'Posición ' + data_name[-1]
                data = np.array(self.data_dictionary[data_name])
                plt.plot(time_data, data, label=new_data_name)
        plt.legend()        

    def plot_all_doppler(self):        
        def low_pass_filter(data, band_limit, sampling_rate):
            cutoff_index = int(band_limit * data.size / sampling_rate)
            F = np.fft.rfft(data)
            F[cutoff_index + 1:] = 0
            return np.fft.irfft(F, n=data.size).real
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Velocidad (cm/s)")
        plt.title("Velocidad vs. tiempo")
        plt.grid()
        time_data = self.data_dictionary['time']
        
        # tracker_pos = np.array(self.data_dictionary['Tracker_position_x'])
        # ideal_speed = np.gradient(-tracker_pos, 1.0/24.0)
        # plt.plot(time_data, ideal_speed, label='Tracker_speed')
        #np.gradient(np.sin(x), dx)
        for data_name in self.data_dictionary['data_names_to_plot']:
            data = np.array(self.data_dictionary[data_name])
            # if data_name.startswith('Doppler_deviation_filtered'):
            #     #plt.plot(time_data, low_pass_filter(data, 3, 24), label=data_name)
            #     plt.plot(time_data, data, label='Velocidad')
            # el
            if 'oppler' in data_name and not 'filtered' in data_name:
                data_name = 'Velocidad en f = ' + data_name.split('_')[-2] + ' Hz'
                plt.plot(time_data, data, label=data_name)
        # left_speaker_crosses = np.array(self.data_dictionary['left_speaker_crosses']) + 1
        # right_speaker_crosses = np.array(self.data_dictionary['right_speaker_crosses']) + 1
        # time_data = self.data_dictionary['time']
        # for data_name in self.data_dictionary['data_names_to_plot']:
        #     data = np.array(self.data_dictionary[data_name])
        #     if data_name.startswith('doppler_deviation_filtered'):
        #         if data_name.endswith('0'):
        #             plt.plot(time_data, data, '-D', label=data_name, markevery=left_speaker_crosses)
        #         elif data_name.endswith('1'):
        #             plt.plot(time_data, data, '-D', label=data_name, markevery=right_speaker_crosses)
        plt.legend(loc='upper right')
            # elif data_name.startswith('predictor'):
            #     plt.plot(time_data, data, label=data_name)
            #     dydx = np.diff(data)/np.diff(time_data)
            #     #plt.plot(time_data, -np.append(dydx, 0), label='derivative')

    def plot_final_doppler(self):        
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Velocidad (cm/s)")
        plt.title("Velocidad vs. tiempo")
        plt.grid()
        time_data = self.data_dictionary['time']
        
        # tracker_pos = np.array(self.data_dictionary['Tracker_position_x'])
        # ideal_speed = np.gradient(-tracker_pos, 1.0/24.0)
        # plt.plot(time_data, ideal_speed, label='Tracker_speed')
        #np.gradient(np.sin(x), dx)
        for data_name in self.data_dictionary['data_names_to_plot']:
            data = np.array(self.data_dictionary[data_name])
            # if data_name.startswith('Doppler_deviation_filtered'):
            #     #plt.plot(time_data, low_pass_filter(data, 3, 24), label=data_name)
            #     plt.plot(time_data, data, label='Velocidad')
            # el
            if 'Doppler_deviation_filtered_' in data_name:
                plt.plot(time_data, data, label='Velocidad')
        # left_speaker_crosses = np.array(self.data_dictionary['left_speaker_crosses']) + 1
        # right_speaker_crosses = np.array(self.data_dictionary['right_speaker_crosses']) + 1
        # time_data = self.data_dictionary['time']
        # for data_name in self.data_dictionary['data_names_to_plot']:
        #     data = np.array(self.data_dictionary[data_name])
        #     if data_name.startswith('doppler_deviation_filtered'):
        #         if data_name.endswith('0'):
        #             plt.plot(time_data, data, '-D', label=data_name, markevery=left_speaker_crosses)
        #         elif data_name.endswith('1'):
        #             plt.plot(time_data, data, '-D', label=data_name, markevery=right_speaker_crosses)
        plt.legend()        
            # elif data_name.startswith('predictor'):
            #     plt.plot(time_data, data, label=data_name)
            #     dydx = np.diff(data)/np.diff(time_data)
            #     #plt.plot(time_data, -np.append(dydx, 0), label='derivative')

    def generate_figure(self):        
        # plt.yticks(np.arange(-60, 60, 5))
        # from mpl_toolkits import mplot3d
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')

        # ax = plt.axes(projection='3d')

        # # Data for a three-dimensional line
        # zline = self.data_dictionary['3d_z']
        # xline = self.data_dictionary['3d_x']
        # yline = self.data_dictionary['3d_y']
        # ax.plot3D(xline, zline, yline, 'gray')

        # Data for three-dimensional scattered points
        # zdata = 15 * np.random.random(100)
        # xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
        # ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
        # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

        #plt.figure(figsize=(8, 6), dpi=80)
        
        self.plot_position()
        plt.figure()
        self.plot_all_doppler()
        plt.figure()
        self.plot_final_doppler()
        #plt.figure()
        #self.plot_position_and_doppler_filtered()

    def add_data(self, name, data, plot=False):
        if plot: 
            self.data_dictionary['data_names_to_plot'].append(name)
        self.data_dictionary[name] = data

    def add_sample(self, name, sample):
        if name not in self.data_dictionary:
            self.data_dictionary[name] = []
            self.data_dictionary['data_names_to_plot'].append(name)
        self.data_dictionary[name].append(sample)        

    def save_to_file(self):
        def serialize(obj):
            if type(obj).__module__ == np.__name__:
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj.item()
            raise TypeError('Unknown type:', type(obj))

        description = input("Enter data description: ")
        self.data_dictionary['description'] = description
        file_path = self.SAVED_DATA_PATH + 'data/' + self.start_timestamp + '.json'
        audio_samples = self.data_dictionary['audio_samples']
        for i, samples in enumerate(audio_samples):
            self.data_dictionary['audio_samples'][i] = samples
        with open(file_path, 'w') as f:
            json.dump(self.data_dictionary, f, default=serialize)

        file_path = self.SAVED_DATA_PATH + 'figures/' + self.start_timestamp + '_' + description + '.png'
        figure = self.generate_figure()
        figure.savefig(file_path)


    def load_from_file(self, folder=None):
        if folder is None:
            folder = self.SAVED_DATA_PATH + 'data/'
        filenames = [file for file in os.listdir(folder)]
        [print(f"[{i}]", filename) for i, filename in enumerate(filenames)]

        path = folder + '/' + filenames[int(input("Enter file index:"))]      
        with open(path, 'r') as file:
            self.data_dictionary = json.load(file)

        for key, value in self.data_dictionary.items():
            if isinstance(value, list):
                self.data_dictionary[key] = np.array(value)

    def compute_metrics(self):
        metrics = {}
        predicted_pos = []
        tracked_pos = []
        for data_name in self.data_dictionary:
            data = self.data_dictionary[data_name]
            if 'Predictor_position' in data_name or 'Predicción posición' in data_name:
                predicted_pos.append(np.array(data, dtype=np.float64))
            elif 'Tracker_position' in data_name or data_name.startswith('Posición'):
                tracked_pos.append(np.array(data))
        if len(tracked_pos) == 0:
            return

        error = abs(np.array(tracked_pos, dtype=np.float64) - np.array(predicted_pos, dtype=np.float64))
        
        avg_error = np.mean(error, axis=1, dtype=np.float64) # Compute error for each coordinate
        coords_names = ['x', 'y', 'z']
        metrics['Total avg error'] = np.mean(avg_error)
        for coord, error in zip(coords_names, avg_error):
            metrics[f'Avg error {coord}'] = error

        # if 'predictor_position_y' in self.data_dictionary:
        #     tracker_position_y = np.array(self.data_dictionary['tracker_position_y'])
        #     predictor_position_y = np.array(self.data_dictionary['predictor_position_y'])
        #     error = np.abs(tracker_position_y[movement_start_time:] - predictor_position_y[movement_start_time:])
        #     avg_error = np.sum(error)/(len(time)-movement_start_time)
        #     metrics['Mean error Y: '] = avg_error

        #     if 'kalman_filter_y' in self.data_dictionary:
        #         kalman_y = self.data_dictionary['kalman_filter_y']
        #         error = np.abs(tracker_position_y[movement_start_time:] - kalman_y[movement_start_time:])
        #         avg_error = np.sum(error)/(len(time)-movement_start_time)
        #         metrics['Kalman error Y: '] = avg_error

        # metrics['Highest error: '] = max(error)

        # if 'kalman_filter_x' in self.data_dictionary:
        #     kalman_x = self.data_dictionary['kalman_filter_x']
        #     error = np.abs(tracker_position_x[movement_start_time:] - kalman_x[movement_start_time:])
        #     avg_error = np.sum(error)/(len(time)-movement_start_time)
        #     metrics['Kalman error X: '] = avg_error
        
        self.metrics = metrics
        return metrics

    def print_metrics(self):
        if self.metrics is None:
            self.metrics = self.compute_metrics()
        print(self.metrics) 

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

    def run_saved(filename=None, folder=None):
        configs = Config.get_all_configs(folder=folder) if filename is None else [Config.read_config(filename=filename, offline=True)]
        options = {'kalman_filter': None, 'constant_dt': None, 'doppler_threshold': { "values": [1, 1.35, 1.5] }, 'outlier_removal': { 'values': [1.35, 1.5, 1.75]}, 'frequency_lookup_width': { 'values': [50, 75, 100] } }
        all_configs = []
        print("Generating all configurations and options combinations...")
        for i in range(len(options) + 1):
            all_i_options_combinations = list(map(dict, itertools.combinations(options.items(), i))) # All options combinations of i elements
            for current_conf, current_opts in list(itertools.product(configs, all_i_options_combinations)): # For each config and options combination
                expanded_values = True
                saved_opts = ujson.loads(ujson.dumps(current_opts))
                while expanded_values: # Ends when no values remain in no node
                    expanded_values = False
                    current_opts_copy = ujson.loads(ujson.dumps(current_opts))
                    current_conf_copy = ujson.loads(ujson.dumps(current_conf))
                    skip = False
                    for key, val in saved_opts.items():  # For each option
                        if val is not None and 'values' in val: # Expand all first nodes with values and delete value, then repeat until none is expanded
                            j = -1
                            for index, val in enumerate(val['values']):
                                if val is not None:
                                    j = index
                                    break
                            if j == -1:
                                skip = True
                            else:
                                current_opts_copy[key]["index"] = j
                                saved_opts[key]['values'][j] = None
                                expanded_values = True
                    #cuando todos son menos -1 se va a insertar el original, ignorarlo
                    if not skip:    
                        current_conf_copy['options'] = current_opts_copy # When finished this iteration, insert config
                        all_configs.append(current_conf_copy)
                        
            
        print("Total combinations:", len(all_configs), "\n")
        print("Starting threads...")
        pool = multiprocessing.Pool(processes=os.cpu_count())
        all_results = pool.map(Plotter.offline_loop, all_configs)    

        grouped_results = itertools.groupby(sorted(all_results, key = lambda r: json.dumps(r[2], sort_keys=True)), key = lambda r: r[2])

        results_info = []
        for key, results_group in grouped_results:
            avg_errors = []
            result_string = ''
            for i, result in enumerate(results_group):
                metrics, description, options = result
                result_string += "Results for " + description + ' ' + str(options) + " = " + str(metrics) + "\n"
                avg_error = metrics['Total avg error']
                avg_errors.append(avg_error)
            total_avg_error = np.mean(avg_errors)
            result_string += 'Total avg error: ' + str(total_avg_error) + '\n'
            results_info.append((total_avg_error, result_string))
        
        results_info.sort(key = lambda r: r[0])

        results_filename = 'offline_results/result_' + time.strftime("%d-%m-%Y_%H-%M-%S") + '.txt'
        with open(results_filename, 'w') as f:
            output_info_text = ''.join([res[1] for res in results_info])
            f.write(output_info_text)
        
        print(f"\nFinished, results written to file {results_filename}")

if __name__ == '__main__':
    plotter = Plotter()
    plotter.load_from_file()
    plotter.print_metrics()
    plotter.plot()
