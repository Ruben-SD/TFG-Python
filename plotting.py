import time
import matplotlib.pyplot as plt
import numpy as np
import os
import json

class Plotter:
    def __init__(self) -> None:
        self.start_timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
        self.data_dictionary = {}
        self.SAVED_DATA_PATH = './saved_data/'
        self.data_dictionary['data_names_to_plot'] = []
        self.metrics = None

    def plot(self):
        self.generate_figure()
        plt.show()

    def generate_figure(self):
        time_data = self.data_dictionary['time']
        plt.xlabel("Time (s)")
        plt.ylabel("Position (cm)")
        plt.title("Position over time")
        plt.yticks(np.arange(-60, 60, 5))
        plt.grid()
        for data_name in self.data_dictionary['data_names_to_plot']:
            data = np.array(self.data_dictionary[data_name])
            if data_name.startswith('tracker_position_'):
                plt.fill_between(time_data, data - 0.5, data + 0.5, label=data_name, facecolor='black')
            elif not data_name.startswith('audio_samples') and not data_name == 'time': 
                plt.plot(time_data, data, label=data_name)
        plt.legend()        
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
        tracker_position_x = np.array(self.data_dictionary['tracker_position_x'])
        predictor_position_x = np.array(self.data_dictionary['predictor_position_x'])
    
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
