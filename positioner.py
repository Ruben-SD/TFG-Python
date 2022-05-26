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
        self.distances = np.array(config['smartphone']['position'], dtype=float)


    def move_by(self, displacements):
        self.distances += displacements

    # def get_other_position(self):
    #     x, y, z = self.get_position()
    #     xR, yR, zR = -x + self.speakers_distance, y
    #     return (xR, yR)

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

        def equations(guess):
            x, y, z, r = guess
            r = 0
            return (
                (x - x1)**2 + (y - y1)**2 + (z - z1)**2 - (dist_1 - r)**2,
                (x - x2)**2 + (y - y2)**2 + (z - z2)**2 - (dist_2 - r)**2,
                (x - x3)**2 + (y - y3)**2 + (z - z3)**2 - (dist_3 - r)**2
            )

        from scipy.optimize import least_squares
        result = least_squares(equations, ((x1+x2+x3)/3, (y1+y2+y3)/3, (z1+z2+z3)/3,
                               (dist_1+dist_2+dist_3)/3), gtol=0.04, ftol=0.04, xtol=0.04)['x']
        (x, y, z) = (result[0], result[1], result[2])
        
        # plotter.add_sample("positioner_distance_left", dL)
        # plotter.add_sample("positioner_distance_right", dR)

        #print(f"DL: {dL}, DR:{dR}")

        return (x, y, z)

import matplotlib.pyplot as plt

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
        x = np.arange(0, 2*np.pi, 0.01)
        self.line, = self.ax.plot(x, np.sin(x))
        def animate(i):
            pos = self.position.distances
            #self.line.set_xdata([pos[0]])  # update the data
            print(pos)
            self.line.set_data(np.arange(100), np.arange(100))
            self.ax.set_xlim(100)
            self.ax.set_ylim(100)
            self.ax.set_zlim(100)
            # self.line.set_ydata([pos[2]])
            self.line.set_3d_properties(np.arange(100))
            return self.line,
        def init():

            return self.line,
        self.ani = animation.FuncAnimation(self.fig, animate, np.arange(1,200), init_func=init, interval=1000/24, blit=False)
        plt.show(block=False)
        # self.ax = plt.axes(projection='3d')
        # self.plot = self.ax.plot([0], [0], [0], 'ro')[0]

    def update(self, dt):
        raise NotImplemented()

    def get_position(self):
        coords = ['x', 'y', 'z']
        position = self.position.distances
        self.plotter.add_sample("3d_x", position[0])
        self.plotter.add_sample("3d_y", position[1])
        self.plotter.add_sample("3d_z", position[2])
        plt.pause(0.05)
        #self.ax.cla()
        # for i, coordinate in enumerate(position):
        #     self.plotter.add_sample(
        #         f"{self.name}_position_{coords[i]}", coordinate)
        return position
