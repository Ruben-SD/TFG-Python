import numpy as np

class Position:
    def __init__(self, config):
        self.distances = np.array(config['smartphone']['distance'], dtype=float)

    def set(self, new_distances):
        self.distances = new_distances

    def move_by(self, displacements):
        raise NotImplemented()

    def get_position(self):
        raise NotImplemented()

    def __add__(self, displacements):
        return self.distances + displacements

class Distance1D(Position):

    def move_by(self, displacements):
        self.distances += np.mean(displacements)

    def get_position(self):
        return self.distances[0]

class Position2D(Position):
    def __init__(self, config):
        self.position = config['smartphone']['position']

    def set(self, new_position):
        self.position = new_position

    def get_position(self):
        return self.position[0], self.position[1]

    def __add__(self, displacements):
        return self.position - np.flip(displacements)

class Distance2D(Position):
    def __init__(self, config):
        position = np.array(config['smartphone']['position'], dtype=float)
        self.speakers_distance = config['speakers_distance']
        self.distances = [np.linalg.norm(position), np.linalg.norm(position - np.array([self.speakers_distance, 0]))]

    def move_by(self, displacements):
        self.distances += displacements

    def get_position(self):
        D = self.speakers_distance
        dL = self.distances[0]
        dR = self.distances[1]
        
        theta = np.arccos((dL*dL + D*D - dR*dR)/(2*D*dL))
        (x, y) = (dL * np.cos(theta), dL * np.sin(theta))
        
        #print(f"DL: {dL}, DR:{dR}, x: {x}, y: {y}, D: {D}")

        return (x, y)


class Positioner:
    def __init__(self, config):
        position_types = {
            '1D': Distance1D,
            '2D': Distance2D
        }
        
        self.position = position_types[config['positioner_type']](config)

    def update(self, dt):
        raise NotImplemented()

    def get_position(self):
        return self.position.get_position()