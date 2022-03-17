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

    def plot(self):
        self.generate_figure()
        plt.show()

    def generate_figure(self):
        time_data = self.data_dictionary['time']
        plt.xlabel("Time (s)")
        plt.ylabel("Position (cm)")
        plt.title("Position over time")
        plt.xticks(np.arange(0, len(time_data), 0.25))
        plt.yticks(np.arange(17990, 18020, 1))
        plt.grid()
        for data_name in self.data_dictionary['data_names_to_plot']:
            data = np.array(self.data_dictionary[data_name])
            if data_name.startswith('tracker_position_'):
                plt.fill_between(time_data, data - 0.5, data + 0.5, label=data_name)
            else: plt.plot(time_data, data, label=data_name)
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

    def print_metrics(self, config):
        real_x_position = np.array(self.data_dictionary['tracker_position_x'])
        predicted_x_position = np.array(self.data_dictionary['predictor_position_x'])
        doppler = np.array(self.data_dictionary['doppler_deviation_filtered_0'])
        time_data = self.data_dictionary['time']
        movement_start_time = next(i for i, d in enumerate(doppler) if d > 4)
        print("Movement starts at: " + str(time_data[movement_start_time]))
        error = np.abs(real_x_position[movement_start_time:] - predicted_x_position[movement_start_time:])
        avgError = np.sum(error)/(len(time_data)-movement_start_time)
        print("Mean error = " + str(avgError))
        print("Highest error = " + str(max(error)))

plotter = Plotter()


if __name__ == '__main__':
    plotter.load_from_file()
    plotter.print_metrics()
    plotter.plot()
