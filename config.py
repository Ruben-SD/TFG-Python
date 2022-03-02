import json
from plotter import *

class Config:
    @staticmethod
    def read_config(filename):
        config = None
        with open(filename, 'r') as file:
            config = json.load(file)
        plotter.add_data('config', config)
        return config