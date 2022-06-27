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
matplotlib.use('Agg')
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
                     'legend.fontsize': 30,
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
                data_name = 'f = ' + data_name.split('_')[-2] + ' Hz'
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
                if data_name.endswith('0'):
                    plt.plot(time_data, data, label='Velocidad respecto al altavoz izquierdo')#, markevery=self.data_dictionary['zcross_l'])
                else:
                    plt.plot(time_data, data, label='Velocidad respecto al altavoz derecho')#, markevery=self.data_dictionary['zcross_r'])
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
        pos = plt.figure()
        self.plot_position()
        plt.figure()
        self.plot_all_doppler()
        plt.figure()
        self.plot_final_doppler()
        #plt.figure()
        #self.plot_position_and_doppler_filtered()
        return pos

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
            return {'Total avg error': -1}

        if len(predicted_pos) == 1:
            error = abs(np.array(tracked_pos, dtype=np.float64) - np.array(predicted_pos, dtype=np.float64))
            avg_error = np.mean(error, axis=1, dtype=np.float64) # Compute error for each coordinate
            coords_names = ['x', 'y', 'z']
            metrics['Total avg error'] = np.mean(avg_error)
            for coord, error in zip(coords_names, avg_error):
                metrics[f'Avg error {coord}'] = error
        else:
            error = np.linalg.norm(np.array(tracked_pos, dtype=np.float64) - np.array(predicted_pos, dtype=np.float64), axis=0)
            avg_error = np.mean(error, dtype=np.float64)
            metrics['Total avg error'] = avg_error
            metrics['Total max error'] = np.max(error)
            metrics['Total std'] = np.std(error)

            error = abs(np.array(tracked_pos, dtype=np.float64) - np.array(predicted_pos, dtype=np.float64))
            avg_error = np.mean(error, axis=1, dtype=np.float64) # Compute error for each coordinate
            max_error = np.max(error, axis=1) # Compute error for each coordinate
            std_error = np.std(error, axis=1) # Compute error for each coordinate
            coords_names = ['x', 'y', 'z']
            for coord, error in zip(coords_names, avg_error):
                metrics[f'Avg error {coord}'] = error
            for coord, error in zip(coords_names, max_error):
                metrics[f'Max error {coord}'] = error
            for coord, error in zip(coords_names, std_error):
                metrics[f'Std error {coord}'] = error

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
        options = {'kalman_filter': None, 'constant_dt': None, 'snr_avg': None, 'doppler_threshold': { "values": [1, 1.35, 1.5] }, 'outlier_removal': { 'values': [1.35, 1.5, 1.75, 2]}, 'frequency_lookup_width': { 'values': [50, 75] } }
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
                    import copy
                    current_conf_copy = copy.copy(current_conf)
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
                        current_conf_copy['options'] = copy.deepcopy(current_opts_copy) # When finished this iteration, insert config
                        all_configs.append(current_conf_copy)
                        
            
        print("Total combinations:", len(all_configs), "\n")
        print("Starting threads...")
        pool = multiprocessing.Pool(processes=os.cpu_count())
        all_results = pool.map(Plotter.offline_loop, all_configs)    

        grouped_results = itertools.groupby(sorted(all_results, key = lambda r: json.dumps(r[2], sort_keys=True)), key = lambda r: r[2])
        time_str = time.strftime("%d-%m-%Y_%H-%M-%S")
        results_info = []
        for key, results_group in grouped_results:
            avg_errors = []
            max_errors = []
            std_errors = []
            avg_errors_x = []
            std_errors_x = []
            max_errors_x = []
            avg_errors_y = []
            std_errors_y = []
            max_errors_y = []
            result_string = ''
            for i, result in enumerate(results_group):
                metrics, description, options = result
                result_string += "Results for " + description + ' ' + str(options) + " = " + str(metrics) + "\n"
                avg_error = metrics['Total avg error']
                max_error = metrics['Total max error']
                std_error = metrics['Total std']
                avg_errors.append(avg_error)
                max_errors.append(max_error)
                std_errors.append(std_error)
                avg_errors_x.append(metrics['Avg error x'])
                std_errors_x.append(metrics['Std error x'])
                max_errors_x.append(metrics['Max error x'])
                avg_errors_y.append(metrics['Avg error y'])
                std_errors_y.append(metrics['Std error y'])
                max_errors_y.append(metrics['Max error y'])

            total_avg_error = np.mean(avg_errors)
            total_max_errors = np.mean(max_errors)
            total_std_errors = np.mean(std_errors)
            total_avg_errors_x = np.mean(avg_errors_x)
            total_std_errors_x = np.mean(std_errors_x)
            total_max_errors_x = np.mean(max_errors_x)
            total_avg_errors_y = np.mean(avg_errors_y)
            total_std_errors_y = np.mean(std_errors_y)
            total_max_errors_y = np.mean(max_errors_y)
            

            original_avg_errors = np.array([3.5278580027815383, 4.790490049601273, 5.560448432995089, 3.6692583243264507, 6.317731869693616, 4.863624150387261,
            4.358759331720689, 4.249078194808773, 5.628888598343067, 9.873530106852574] + [5.283966706151032])
            original_max_errors = np.array([10.786126366927565, 13.33180988045995, 11.664472228572698, 10.382514760304803, 15.805127835445942, 12.747257820232589,
            11.782980813555348, 16.30153743411097, 17.921600025970662, 15.628099567121568] + [13.635152673270209])
            original_std_errors = np.array([2.8910836797250496,  3.204434180441297, 3.179669684866274, 2.1766703483377468, 3.754848070383608, 2.7174188823403846, 
            2.7085495517554072, 2.86336930921281, 4.080949654179546, 2.6600334008670745] + [3.0237026762109194])
            original_avg_errors_x = np.array([2.1076872124190675, 3.03364106506349, 1.9714553089749678, 2.739893832737707, 2.8710478384470486, 3.552404751977964,
            2.745872500648192, 2.236138213863272, 2.8427915716670946, 3.961865101913661] + [2.8062797397712465])
            original_std_errors_x = np.array([2.224685729249608, 2.483472201725879, 1.5251291737355592, 2.1547545331151103, 2.645463568866009, 2.5960225829468624,
            2.08001051559044, 1.9831032829529573, 2.947460031408139, 2.614300528833436] + [2.3254402148424])
            original_max_errors_x = np.array([8.843813083405017, 8.947908973117727, 7.5980981283296245, 10.33824093767214, 10.67125038137398, 11.143023847681924,
            9.04768027562561, 13.028809802715237, 13.072543711421744, 11.753911248474623] + [10.444528038981762])
            original_avg_errors_y = np.array([2.588846143414582, 2.9314015409781065, 4.944438442609387, 1.9283091052124939, 5.318773755183007, 2.7221709180489144,
            2.8972611234027195, 3.3618095445288043, 4.535595594351837, 8.700938112904778] + [3.992954428063463])
            original_std_errors_y = np.array([2.170357251092087, 3.041951026771428, 3.220055828124211, 1.5274125631511837, 3.2375720170111433, 2.0664167000421974,
            2.4646788906900334, 2.453318803274013, 3.3163119546132886, 2.514976755207402] + [2.601305178997699])
            original_max_errors_y = np.array([8.000598638112898, 13.323682831185991, 11.3186686412635, 6.571306302352667, 12.100591875709028, 7.623575911823892,
            8.42655555984578, 11.47421293545515, 12.278353879914363, 13.434061438973004] + [10.455160801463627])

            Plotter.save_fig('avg_error', time_str, key, np.array(avg_errors + [total_avg_error]), original_avg_errors, 'Error medio (cm)')
            Plotter.save_fig('max_error', time_str, key, np.array(max_errors + [total_max_errors]), original_max_errors, 'Error máximo (cm)')
            Plotter.save_fig('std_error', time_str, key, np.array(std_errors + [total_std_errors]), original_std_errors, 'Desviación típica (cm)')
            Plotter.save_fig('avg_error_x', time_str, key, np.array(avg_errors_x + [total_avg_errors_x]), original_avg_errors_x, 'Error medio en el eje X (cm)')
            Plotter.save_fig('std_error_x', time_str, key, np.array(std_errors_x + [total_std_errors_x]), original_std_errors_x, u'\u03C3' + ' en el eje X (cm)')
            Plotter.save_fig('max_error_x', time_str, key, np.array(max_errors_x + [total_max_errors_x]), original_max_errors_x, 'Error máximo en el eje X (cm)')
            Plotter.save_fig('avg_error_y', time_str, key, np.array(avg_errors_y + [total_avg_errors_y]), original_avg_errors_y, 'Error medio en el eje Y (cm)')
            Plotter.save_fig('std_error_y', time_str, key, np.array(std_errors_y + [total_std_errors_y]), original_std_errors_y, u'\u03C3' + ' en el eje Y (cm)')
            Plotter.save_fig('max_error_y', time_str, key, np.array(max_errors_y + [total_max_errors_y]), original_max_errors_y, 'Error máximo en el eje Y (cm)')


            result_string += f'Avg error: {total_avg_error} Max error: {total_max_errors} Std error: {total_std_errors} Avg error x: {total_avg_errors_x} Std error x: {total_std_errors_x} Max error x: {total_max_errors_x} Avg error y: {total_avg_errors_y} Std error y: {total_std_errors_y} Max error y: {total_max_errors_y}' +  '\n'
            results_info.append((total_avg_error, result_string))
        
        results_info.sort(key = lambda r: r[0])

        results_filename = '2D Results/result_' + time_str + '.txt'
        with open(results_filename, 'w') as f:
            output_info_text = ''.join([res[1] for res in results_info])
            f.write(output_info_text)
        
        print(f"\nFinished, results written to file {results_filename}")

    @staticmethod
    def save_fig(name, time_str, options, data, original, ylabel):

        options_hash = hash(tuple(options))

        x = np.arange(len(data))
        fig, ax = plt.subplots()
        ax.bar(x - 0.2, original, width=0.4, label = 'Original')
        ax.bar(x + 0.2, data, width=0.4, label = 'Con filtros')
        #splt.xticks(distances, distances)
        ax.set_xlabel('Número de prueba')  # Número de prueba
        ax.set_ylabel(ylabel)
        #ax.set_yticks(np.arange(0, max(max(data), max(original_avg_errors)), 1))
        ax.legend()
        ax.set_xticks(x, [i for i in range(len(x) - 1)] + [u'\u0078\u0304'])
        ax.set_title('Error del posicionamiento en cada prueba')
        ax.grid()
        from pathlib import Path
        folder = f'Figures_{time_str}/{options_hash}/'
        Path(folder).mkdir(parents=True, exist_ok=True)
        filename = f'{folder}{name}.pdf'
        fig.savefig(filename, bbox_inches='tight')
        f = open(f'Figures_{time_str}/{options_hash}/options.txt', "a")
        f.write(json.dumps(options, indent=4, sort_keys=True, default=str) + '\n\n')
        f.close()
        plt.close(fig)


if __name__ == '__main__':
    plotter = Plotter()
    plotter.load_from_file()
    plotter.print_metrics()
    plotter.plot()
