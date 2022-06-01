from turtle import pos, position
from matplotlib import animation, projections
import numpy as np
import plotting


class Position:
    def __init__(self, config):
        self.distances = np.array(
            config['smartphone']['distance'], dtype=float)

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
        dR = np.sqrt((self.speakers_distance - x) *
                     (self.speakers_distance - x) + y * y)
        # plotter.add_sample("tracker_distance_left", dL)
        # plotter.add_sample("tracker_distance_right", dR)
        return self.position[0], self.position[1]

    def __add__(self, displacements):
        return self.position - np.flip(displacements)


class Distance2D(Position):
    def __init__(self, config):
        position = np.array(config['smartphone']['position'], dtype=float)
        self.speakers_distance = config['speakers_distance']
        self.distances = [np.linalg.norm(position), np.linalg.norm(
            position - np.array([self.speakers_distance, 0]))]

    def move_by(self, displacements):
        self.distances += displacements

    def get_other_position(self):
        x, y = self.get_position()
        xR, yR = -x + self.speakers_distance, y
        return (xR, yR)

    def get_position(self):
        D = self.speakers_distance
        dL = self.distances[0]
        dR = self.distances[1]
        # plotter.add_sample("positioner_distance_left", dL)
        # plotter.add_sample("positioner_distance_right", dR)
        theta = np.arccos((dL*dL + D*D - dR*dR)/(2*D*dL))
        (x, y) = (dL * np.cos(theta), dL * np.sin(theta))

        #print(f"DL: {dL}, DR:{dR}")

        return (x, y)


class Distance3D(Position):
    def __init__(self, config):
        self.speakers_pos = [np.array(speakerConfig['pos'], dtype=float)
                             for speakerConfig in config['speakers']]
        coords = np.array(config['smartphone']['position'], dtype=float)
        self.distances = [np.linalg.norm(speaker_pos - coords) for speaker_pos in self.speakers_pos] # Distances from phone to each speaker
        
        self.last_prediction = coords

    def move_by(self, displacements):
        #print(displacements)
        self.distances += displacements

    # def get_other_position(self):
    #     x, y, z = self.get_position()
    #     xR, yR, zR = -x + self.speakers_distance, y
    #     return (xR, yR)

    def gps_solve(self, distances_to_station, stations_coordinates):
        def error(x, c, r):
            # calcular distancia respecto al ultimo punto predicho
            return sum([(np.linalg.norm(x - c[i]) - r[i]) ** 2 for i in range(len(c))] + [np.linalg.norm((x - self.last_prediction))])

        l = len(stations_coordinates)
        S = sum(distances_to_station)
        # compute weight vector for initial guess
        W = [((l - 1) * S) / (S - w) for w in distances_to_station]
        # get initial guess of point location
        x0 = sum([W[i] * stations_coordinates[i] for i in range(l)])
        # optimize distance from signal origin to border of spheres
        return minimize(error, self.last_prediction, args=(stations_coordinates, distances_to_station), method='Nelder-Mead').x

    def get_position(self):

        dL = self.distances[0]
        dR = self.distances[1]
        dX = self.distances[2]

        # return (dL, dR, dX)

        pos0 = self.speakers_pos[0]
        pos1 = self.speakers_pos[1]
        pos2 = self.speakers_pos[2]

        x1, y1, z1, dist_1 = (pos0[0], pos0[1], pos0[2], dL)
        x2, y2, z2, dist_2 = (pos1[0], pos1[1], pos1[2], dR)
        x3, y3, z3, dist_3 = (pos2[0], pos2[1], pos2[2], dX)
        
        # stations = list(np.array([[0,0,0], [40,0,0], [70,22.5,60], [70,22.5,60]]))
        # distances_to_station = [30.103986447, 30.103986447, 78.1024967591, 78.1024967591]
        stations = list(np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x3, y3, z3]]))
        distances_to_station = [dist_1, dist_2, dist_3, dist_3]
        self.last_prediction = self.gps_solve(distances_to_station, stations)
        return self.last_prediction
        # plotter.add_sample("positioner_distance_left", dL)
        # plotter.add_sample("positioner_distance_right", dR)

        #print(f"DL: {dL}, DR:{dR}")

from scipy.optimize import minimize
import numpy as np

import matplotlib.pyplot as plt
import time

class Positioner:
    def __init__(self, config, plotter):
        self.plotter = plotter

        position_types = {
            '1D': Distance1D,
            '2D': Distance2D,
            '3D': Distance3D
        }

        self.position = position_types[config['positioner_type']](config)

        

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.li, = self.ax.plot(0,0 )
        #text = self.fig.text(0, 1, "TEXT", va='top') 

        # def update(i):
        #     pos = self.plotter.data_dictionary            
        #     if '3d_x' in pos:
        #         self.graph._offsets3d = ([pos['3d_x'][-1]], [pos['3d_z'][-1]], [pos['3d_y'][-1]])
        #     return self.graph,
        self.ax.set_xlim3d(-20, 150)
        self.ax.set_ylim3d(-20, 200)
        self.ax.set_zlim3d(-20, 75)

        # self.ani = animation.FuncAnimation(self.fig, update, frames=200, interval=50, blit=False)
        plt.show(block=False)
        self.last = time.time()
        # self.ax = plt.axes(projection='3d')
        # self.plot = self.ax.plot([0], [0], [0], 'ro')[0]

    def update(self, dt):
        raise NotImplemented()

    def get_position(self):
        coords = ['x', 'y', 'z']
        position = self.position.get_position()
        self.plotter.add_sample("3d_x", position[0])
        self.plotter.add_sample("3d_y", position[1])
        self.plotter.add_sample("3d_z", position[2])

        
        plt.pause(0.001)
        if time.time() - self.last > 0.15:
            pos = self.plotter.data_dictionary
            self.li.set_data(pos['3d_x'][-15:], pos['3d_y'][-15:])#, pos['3d_z'])
            self.li.set_3d_properties(pos['3d_z'][-15:])
            self.fig.canvas.draw()
            self.last = time.time()
        #self.ax.cla()
        # for i, coordinate in enumerate(position):
        #     self.plotter.add_sample(
        #         f"{self.name}_position_{coords[i]}", coordinate)
        return position, self.position.distances
