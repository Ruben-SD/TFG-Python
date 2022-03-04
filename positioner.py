from turtle import position
import numpy as np

class Position2D:
    def __init__(self, init) -> None:
        self.x = init['x']
        self.y = init['y']

class Distance:
    def __init__(self, init) -> None:
        self.distance = init['distance']

class Positioner:
    def __init__(self, config):
        distances_config = config['smartphone']['distance']
        self.two_speakers = len(config['speakers']) == 2
        self.two_dimensions = config['2d'] if '2d' in config else False
        self.speakers_distance = config['speakers_distance'] if 'speakers_distance' in config else None
        self.initial_distance = np.array([distance for distance in distances_config], dtype=float)        
        self.distances = self.initial_distance.copy()
        position_data_types = { 'distance': Distance,
                                'position2D': Position2D
                              }
        positions_data_config = config['positions_data']
        self.position_data = [position_data_types[position_data_config['type']](position_data_config['init']) for position_data_config in positions_data_config]

    def set_distance(self, new_distance):
        self.distances = new_distance

    def move_by(self, displacements):
        self.distances -= displacements

    def get_distance(self):
        return self.distances

    def get_position(self):
        if self.two_dimensions:
            D = self.speakers_distance
            dL = self.distances[0]
            dR = self.distances[1]
            
            theta = np.arccos((dL*dL + D*D - dR*dR)/(2*D*dL))
            (x, y) = (dL * np.cos(theta), dL * np.sin(theta))
            #print(f"DL: {dL}, DR:{dR}, x: {x}, y: {y}, D: {D}")
            return (x, y)
        else:
            return self.distances
        

    def update_position(self, dt):
        pass