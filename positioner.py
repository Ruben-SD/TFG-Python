import numpy as np


class Vector2D:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

class Vector3D:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z




class Positioner:
    def __init__(self, config):
        position_configs = config['smartphone']['position'] 
        distance_configs = config['smartphone']['distance']
        # TODO dynamically change between 1d, 2d or 3d?
        
        #the same but for n speakers, two dimensional
        self.initial_distance = np.array([[distance for _, distance in distance_config.items()] for distance_config in distance_configs], dtype=float)
        self.initial_position = np.array([[position for _, position in position_config.items()] for position_config in position_configs], dtype=float)
        self.position = self.initial_position.copy()

    def set_position(self, new_position):
        self.position = self.initial_position - new_position

    def move_by(self, amounts):
        self.position -= amounts

    def get_distance(self):
        return self.initial_distance + self.position - self.initial_position

    def update_position(self, dt):
        pass