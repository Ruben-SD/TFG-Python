import json
from plotter import *

class Config:
    CONFIGS_PATH='configs/'

    @staticmethod
    def read_config(filename=None):
        config = None
        if filename is None:
            filename = Config.ask_for_filename()
        with open(Config.CONFIGS_PATH + filename, 'r') as file:
            config = json.load(file)
        plotter.add_data('config', config)
        return config

    @staticmethod
    def ask_for_filename():
        filenames = [file for file in os.listdir(Config.CONFIGS_PATH)]
        [print(f"[{i}]", filename) for i, filename in enumerate(filenames)]

        filename = filenames[int(input("Enter file index: "))]      
        return filename