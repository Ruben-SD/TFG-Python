import json
from plotter import *

with open('config.json', 'r') as file:
    config = json.load(file)
plotter.add_data('config', config)