import json
import os
import sys

class Config:
    CONFIGS_PATH = sys.path[0] + '/configs/'
    SAVED_DATA_PATH = sys.path[0] + '/saved_data/data/'
    OFFLINE_DATA_PATH = sys.path[0] + '/Memoria/1D/'

    @staticmethod
    def read_config(index=None, filename=None, offline=False):
        folder = Config.CONFIGS_PATH if not offline else Config.SAVED_DATA_PATH

        if filename is None:
            filename = Config.ask_for_filename(folder, index)

        with open(folder + filename, 'r') as file:
            config = json.load(file)
        config['offline'] = offline
        config['options'] = {}

        return config

    @staticmethod
    def ask_for_filename(folder, index=None):
        filenames = [file for file in os.listdir(folder)]

        if index is not None:
            return filenames[index]

        [print(f"[{i}]", filename) for i, filename in enumerate(filenames)]
        filename = filenames[int(input("Enter file index: "))]
        return filename

    @staticmethod
    def get_all_configs(index=None):
        configs = []
        folder = Config.OFFLINE_DATA_PATH
        filenames = [file for file in os.listdir(folder)]
        for filename in filenames:
            with open(folder + filename, 'r') as file:
                config = json.load(file)
                config['offline'] = True
                configs.append(config)
        return configs
