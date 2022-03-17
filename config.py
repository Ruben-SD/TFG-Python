import json
from plotter import *

class Config:
    CONFIGS_PATH='configs/'

    @staticmethod
    def read_config(index=None, filename=None):
        config = None
        if filename is None:
            filename = Config.ask_for_filename(index)
        with open(Config.CONFIGS_PATH + filename, 'r') as file:
            config = json.load(file)
        plotter.add_data('config', config)
        return config

    @staticmethod
    def ask_for_filename(index=None):
        filenames = [file for file in os.listdir(Config.CONFIGS_PATH)]
        
        if index is not None:
            return filenames[index]

        [print(f"[{i}]", filename) for i, filename in enumerate(filenames)]
        filename = filenames[int(input("Enter file index: "))]      
        return filename