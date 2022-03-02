import json
from plotter import *

with open('config_1d_1speaker_18_19.json', 'r') as file:
    config = json.load(file)
plotter.add_data('config', config)