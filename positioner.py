import numpy as np
import plotting
from scipy.optimize import minimize

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
        return self.distances

class Position2D(Position):
    def __init__(self, config):
        self.position = config['smartphone']['position']
        self.speakers_distance = config['speakers_distance']

    def set(self, new_position):
        self.position = new_position

    def get_position(self):
        x, y = self.position[0], self.position[1]
        dL = np.sqrt(x * x + y * y)
        dR = np.sqrt((self.speakers_distance - x) * (self.speakers_distance - x) + y * y)
        # plotter.add_sample("tracker_distance_left", dL)
        # plotter.add_sample("tracker_distance_right", dR)
        return self.position[0], self.position[1]

    def __add__(self, displacements):
        return self.position - np.flip(displacements)

class Distance2D(Position):
    def __init__(self, config):
        position = np.array(config['smartphone']['position'], dtype=float)
        self.speakers_distance = config['speakers_distance']
        self.distances = [np.linalg.norm(position), np.linalg.norm(position - np.array([self.speakers_distance, 0]))]
        self.last_prediction = np.array([20, 23])

    def move_by(self, displacements):
        self.distances += displacements

    def get_other_position(self):
        x, y = self.get_position()
        xR, yR = -x + self.speakers_distance, y
        return (xR, yR)

    def gps_solve(self, distances_to_station, stations_coordinates):
        def error(x, c, r):
            # calcular distancia respecto al ultimo punto predicho
            return sum([(np.linalg.norm(x - c[i]) - r[i]) ** 2 for i in range(len(c))])

        l = len(stations_coordinates)
        S = sum(distances_to_station)
        # compute weight vector for initial guess
        W = [((l - 1) * S) / (S - w) for w in distances_to_station]
        # get initial guess of point location
        x0 = sum([W[i] * stations_coordinates[i] for i in range(l)])
        # optimize distance from signal origin to border of spheres
        return minimize(error, self.last_prediction, args=(stations_coordinates, distances_to_station), method='Nelder-Mead').x

    def get_position(self):
        D = self.speakers_distance
        dL = self.distances[0]
        dR = self.distances[1]
        # plotter.add_sample("positioner_distance_left", dL)
        # plotter.add_sample("positioner_distance_right", dR)
        # theta = np.arccos((dL*dL + D*D - dR*dR)/(2*D*dL))
        # (x, y) = (dL * np.cos(theta), dL * np.sin(theta))

        self.last_prediction = self.gps_solve([dL, dR], list(np.array([[0, 0], [40, 0]])))
        #self.distances = [np.linalg.norm(self.last_prediction), np.linalg.norm(self.last_prediction - np.array([self.speakers_distance, 0]))]
        return self.last_prediction
        
        #print(f"DL: {dL}, DR:{dR}")

        #return (x, y)


class Positioner:
    def __init__(self, config, plotter):
        self.plotter = plotter

        position_types = {
            '1D': Distance1D,
            '2D': Distance2D
        }
        
        self.position = position_types[config['positioner_type']](config)

    def update(self, dt):
        raise NotImplemented()

    def get_position(self):
        coords = ['x', 'y', 'z']
        position = self.position.get_position()
        
        for i, coordinate in enumerate(position):
            self.plotter.add_sample(f"{self.name}_position_{coords[i]}", coordinate)
        return position, self.position.distances