import json
from plotter import *

with open('config_2d_2speakers.json', 'r') as file:
    config = json.load(file)
plotter.add_data('config', config)