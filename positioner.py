from scipy.optimize import minimize
import numpy as np


class Positioner:
    def __init__(self, name, config, plotter):
        self.name = name
        self.plotter = plotter
        self.speakers_pos = [np.array(speaker_config['pos'], dtype=float)
                             for speaker_config in config['speakers']]
        self.position = np.array(config['smartphone']['position'], dtype=float)
        self.update_distances()

    def move_by(self, displacements):
        self.distances += displacements
        self.position = self.multilaterate(
            self.distances, self.speakers_pos, self.position) if len(self.position) != 1 else self.distances
        #self.update_distances()

    def set_position(self, position):
        self.position = position
        self.update_distances()

    def get_position(self):
        return list(self.position), list(self.distances)

    def update_distances(self):
        self.distances = [np.linalg.norm(speaker_pos - self.position) for speaker_pos in self.speakers_pos]

    def update(self, dt):
        self.update_plotting()

    def update_plotting(self):
        coords_names = ['x', 'y', 'z']
        for i, coord in enumerate(self.position): 
            self.plotter.add_sample(self.name + f'_position_{coords_names[i]}', coord)

    def print_position(self):
        print(f"{self.name} position: {self.get_position()}")

    @staticmethod
    def multilaterate(distances_to_stations, stations_coordinates, last_position):
        def error(guess, coords, distances):
            return sum([(np.linalg.norm(guess - coords[i]) - distances[i]) ** 2 for i in range(len(coords))])# + [np.linalg.norm((guess - last_position))])

        # optimize distance from signal origin to border of spheres
        return minimize(error, last_position, args=(stations_coordinates, distances_to_stations), method='Nelder-Mead').x

    def stop(self):
        pass