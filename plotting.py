import time
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json

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
            if 'position' in data_name:
                data = np.array(self.data_dictionary[data_name])
                plt.plot(time_data, data, label=data_name)
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
        for data_name in self.data_dictionary['data_names_to_plot']:
            data = np.array(self.data_dictionary[data_name])
            if data_name.startswith('Doppler_deviation_filtered'):
                #plt.plot(time_data, low_pass_filter(data, 3, 24), label=data_name)
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

        self.plot_position()
        plt.figure()
        self.plot_all_doppler()
        #plt.figure()
        #self.plot_position_and_doppler_filtered()
        
        
        figure = plt.gcf()
        return figure

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


    def load_from_file(self):
        filenames = [file for file in os.listdir(self.SAVED_DATA_PATH + 'data/')]
        [print(f"[{i}]", filename) for i, filename in enumerate(filenames)]

        path = self.SAVED_DATA_PATH + 'data/' + filenames[int(input("Enter file index:"))]      
        with open(path, 'r') as file:
            self.data_dictionary = json.load(file)

        for key, value in self.data_dictionary.items():
            if isinstance(value, list):
                self.data_dictionary[key] = np.array(value)

    def compute_metrics(self):
        metrics = {}
        pos = []
        tracker_position_x = np.array(self.data_dictionary['tracker_position_x'])
        predictor_position_x = np.array(self.data_dictionary['Predictor_position_x'])
    
        if 'doppler_deviation_filtered_0' in self.data_dictionary:
            doppler_deviation_filtered = np.array(self.data_dictionary['doppler_deviation_filtered_0'])
        else:
            doppler_deviation_filtered = np.array(self.data_dictionary['doppler_deviation_filtered_1'])
        time = self.data_dictionary['time']

        movement_start_time = next(i for i, d in enumerate(doppler_deviation_filtered) if d > 4)
        metrics['Movement start time: '] = time[movement_start_time]

        error = np.abs(tracker_position_x[movement_start_time:] - predictor_position_x[movement_start_time:])
        avg_error = np.sum(error)/(len(time)-movement_start_time)
        metrics['Mean error X: '] = avg_error

        if 'predictor_position_y' in self.data_dictionary:
            tracker_position_y = np.array(self.data_dictionary['tracker_position_y'])
            predictor_position_y = np.array(self.data_dictionary['predictor_position_y'])
            error = np.abs(tracker_position_y[movement_start_time:] - predictor_position_y[movement_start_time:])
            avg_error = np.sum(error)/(len(time)-movement_start_time)
            metrics['Mean error Y: '] = avg_error

            if 'kalman_filter_y' in self.data_dictionary:
                kalman_y = self.data_dictionary['kalman_filter_y']
                error = np.abs(tracker_position_y[movement_start_time:] - kalman_y[movement_start_time:])
                avg_error = np.sum(error)/(len(time)-movement_start_time)
                metrics['Kalman error Y: '] = avg_error

        metrics['Highest error: '] = max(error)

        if 'kalman_filter_x' in self.data_dictionary:
            kalman_x = self.data_dictionary['kalman_filter_x']
            error = np.abs(tracker_position_x[movement_start_time:] - kalman_x[movement_start_time:])
            avg_error = np.sum(error)/(len(time)-movement_start_time)
            metrics['Kalman error X: '] = avg_error
        
        self.metrics = metrics
        return metrics

    def print_metrics(self):
        if self.metrics is None:
            self.metrics = self.compute_metrics()
        print(self.metrics) 


if __name__ == '__main__':
    plotter = Plotter()
    plotter.load_from_file()
    plotter.print_metrics()
    plotter.plot()
