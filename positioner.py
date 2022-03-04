from turtle import pos, position
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

    def set_position_data(self, position_data):
        self.position_data = position_data

    def move_by(self, displacements):
        self.distances -= displacements

    def get_distance(self):
        return self.distances

    def get_position_data(self):
        return self.position_data
        #     D = self.speakers_distance
        #     dL = self.distances[0]
        #     dR = self.distances[1]
            
        #     theta = np.arccos((dL*dL + D*D - dR*dR)/(2*D*dL))
        #     (x, y) = (dL * np.cos(theta), dL * np.sin(theta))
        #     #print(f"DL: {dL}, DR:{dR}, x: {x}, y: {y}, D: {D}")
        #     return (x, y)
        # else:
        #     return self.distances


    def set_position_data(self, speeds):
        self.position_data.update(speeds) #y esta position yave q hace dependiendo del tipo q sea



    def update_position(self, dt):
        self.get_new_position_data()
        self.set_position_data()
        return self.get_position_data()

        distancia con 1 altavoz
dos distancias con 2 altavoces
distancia (promediando las dos) con 2 altavoces
posición en 2d
