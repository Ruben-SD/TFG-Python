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
        self.data_names_to_plot = []

    def plot(self):
        plt.xlabel("Time (s)")
        plt.ylabel("Position (cm)")
        plt.title("Position over time")
        plt.grid()
        time_data = self.data_dictionary['time']
        for data_name in self.data_names_to_plot:
            data = self.data_dictionary[data_name]
            plt.plot(time_data, data)
        plt.show()

    def add_data(self, name, data, plot=False):
        if plot: 
            self.data_names_to_plot.append(name)
        self.data_dictionary[name] = data

    def add_sample(self, name, sample):
        self.data_dictionary[name].append(sample)        

    def save_to_file(self):
        def serialize(obj):
            if type(obj).__module__ == np.__name__:
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj.item()
            raise TypeError('Unknown type:', type(obj))
        input()
        self.data_dictionary['description'] = input("Enter data description: ")
        file_path = self.SAVED_DATA_PATH  + self.start_timestamp + '.json'
        with open(file_path, 'w') as f:
            json.dump(self.data_dictionary, f, default=serialize)

    def load_from_file(self):
        filenames = [file for file in os.listdir('./saved_data/')]
        [print(f"[{i}]", filename) for i, filename in enumerate(filenames)]

        path = self.SAVED_DATA_PATH + filenames[int(input("Enter file index:"))]      
        with open(path, 'r') as file:
            self.data_dictionary = json.load(file)

        for key, value in self.data_dictionary.items():
            if isinstance(value, list):
                self.data_dictionary[key] = np.array(value)
        print(self.data_dictionary)

plotter = Plotter()