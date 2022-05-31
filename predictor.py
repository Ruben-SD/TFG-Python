from pykalman import KalmanFilter
import pygame
from positioner import Positioner
from speaker import PhoneSpeaker, Speaker, SpeakersOrchestrator
from receiver import Receiver
from doppleranalyzer import DopplerAnalyzer
import numpy as np


class SpeakerDistanceFinder:
    def __init__(self, plotter):
        self.distances = np.zeros(2)
        self.time = 0
        self.is_on_left = True
        self.delay = False
        self.all_distances = []
        self.all_speeds = []
        self.plotter = plotter
        self.dts= []
        
        self.pos = 'unknown'
        self.speeds_left = []
        self.speeds_right = []
        self.speeds_left_dt = []
        self.speeds_right_dt = []
        self.on_left_speaker = True

    def print_times(self, times):
        for x in times:
            print(np.sum(self.dts[:x]), end=', ')
        print("\n")

    def update(self, dt, displacements):
        self.time += dt
        self.dts.append(dt)

        self.speeds_left.append(displacements[0])
        self.speeds_right.append(displacements[1])

        self.speeds_left_dt.append(displacements[0] * dt)
        self.speeds_right_dt.append(displacements[1] * dt)

        zero_crosses_l = np.where(np.diff(np.signbit(self.speeds_left)))[0][1:]
        left_speaker_crosses = zero_crosses_l[[np.where([np.sum(self.speeds_left[val-20:val]) > 40 and np.abs(val - zero_crosses_l[idx - 1]) > 15 for idx, val in enumerate(zero_crosses_l)])[0]]]

        self.plotter.add_data('zero_crosses_l', zero_crosses_l)
        self.plotter.add_data('left_speaker_crosses', left_speaker_crosses)

        zero_crosses_r = np.where(np.diff(np.signbit(self.speeds_right)))[0][1:]
        right_speaker_crosses = zero_crosses_r[[np.where([np.sum(self.speeds_right[val-20:val]) > 40 and np.abs(val - zero_crosses_r[idx - 1]) > 15 for idx, val in enumerate(zero_crosses_r)])[0]]]

        self.plotter.add_data('zero_crosses_r', zero_crosses_r)
        self.plotter.add_data('right_speaker_crosses', right_speaker_crosses)

        print(left_speaker_crosses)
        print(right_speaker_crosses)
        
        total_distance = 0
        for i, left_speaker_cross in enumerate(left_speaker_crosses[:-1]):
            distance = np.sum(self.speeds_left_dt[left_speaker_cross:right_speaker_crosses[i + 1]])
            print(distance)
            total_distance += distance
        if len(left_speaker_crosses) > 1:
            computed_distance = total_distance/(len(left_speaker_crosses) - 1)
            print("Total distance: ", computed_distance)

        print("SUM=", np.sum(self.speeds_left_dt[64:92]))

        # if len(zero_crossings_l) == 2 and len(zero_crossings_l) == len(zero_crossings_r):
        #     self.on_left_speaker = False

        #     print("On right speaker")
        #     zero_crossings_l = zero_crossings_l[1:]
        #     zero_crossings_r = zero_crossings_r[1:]
        #     print(zero_crossings_l)
        #     print(zero_crossings_r)
        #     distance = np.sum(self.speeds_left[zero_crossings_l[0]:zero_crossings_r[0]])
        #     print("Distance: ", distance)

        #     self.speeds_left = []
        #     self.speeds_right = []
        # elif len(zero_crossings_r) == 2 and len(zero_crossings_r) == len(zero_crossings_l):
        #     print("On left speaker")
        #     self.on_left_speaker = True

        # elif self.on_left_speaker:
        #     # print("On left")
        #     print(zero_crossings_l)
        #     print(zero_crossings_r, "r")
        # else: 
        #     print("On right")
        


        # if len(self.all_speeds) > 100:
            
        #     l, r = map(np.array, zip(*self.all_speeds))
            
        #     zero_crossings_l = np.where(np.diff(np.signbit(l)))[0][1:]
        #     zero_crossings_r = np.where(np.diff(np.signbit(r)))[0][1:]
        #     self.plotter.add_data('zcross_l', zero_crossings_l)
        #     self.plotter.add_data('zcross_r', zero_crossings_r)
  
        #     self.print_times(zero_crossings_l)
        #     self.print_times(zero_crossings_r)

        #     inner_left_crosses = []
        #     for zcross in zero_crossings_l:
        #         if zcross <= 20:
        #             continue
        #         if (l[zcross - 10: zcross] > 0).sum() >= 3: # se estaba acercando? (en el medio)
        #           inner_left_crosses.append(zcross)
                  
        #     self.print_times(inner_left_crosses)
            

        #     inner_right_crosses = []
        #     for zcross in zero_crossings_r:
        #         if zcross <= 20:
        #             continue
        #         if (r[zcross - 10: zcross] > 0).sum() >= 3: # se estaba acercando? (en el medio)
        #           inner_right_crosses.append(zcross)

        #     self.print_times(inner_right_crosses)

        #     total_d = 0
        #     for i, inner_left_cross in enumerate(inner_left_crosses[:-5]):
        #         d = np.sum(l[inner_left_cross:inner_right_crosses[i]])
        #         total_d += d
        #     computed_distance = total_d/len(inner_left_crosses)
        #     print("TOTAL AVG distance = ", computed_distance)
        # if self.time > 0.5 and self.is_on_left and displacements[0] > 0 and self.distances[0] > 5:
        #     self.time = 0            
        #     distance = self.distances[0]
        #     self.all_distances.append(distance)
        #     print(distance)
        #     self.distances = displacements
        #     self.is_on_left = False
        # if self.time > 0.5 and not self.is_on_left and displacements[1] > 0 and self.distances[1] > 5:
        #     self.time = 0
        #     distance = self.distances[1]
        #     self.all_distances.append(distance)
        #     print(distance)
        #     self.distances = displacements
        #     self.is_on_left = True 
        # print("Distance: ", np.mean(self.all_distances))

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
class Predictor(Positioner):
    def __init__(self, config, plotter):
        super().__init__(config, plotter)
        self.name = "predictor"
        
        self.speakers_orchestrator = SpeakersOrchestrator(config)
        self.spakers = self.speakers_orchestrator.get_speakers()
        
        self.receiver = Receiver()
        self.doppler_analyzers = [DopplerAnalyzer(speaker.get_config().get_frequencies(), plotter, config) for speaker in self.speakers]

        self.speaker_distance_finder = SpeakerDistanceFinder(plotter)
        # for i, speaker in enumerate(self.speakers):
        #     plotter.add_data(f'predicted_x_position_{i}', [], plot=True)
        
        # if self.two_dimensions:
        #     plotter.add_data('predicted_y_position', [], plot=True)

        # frequencies = []
        # for speaker in self.speakers:
        #     frequencies = frequencies + speaker.get_config().get_frequencies()
        # for frequency in frequencies: 
        #     plotter.add_data(f'doppler_deviation_{frequency}_hz', [], plot=True)
        
        # plotter.add_data(f'doppler_deviation_chosen', [], plot=True)
        self.my_filter = KalmanFilter(dim_x=2, dim_z=1)
        self.my_filter.x = np.array([[0.],
                [0.]])       # initial state (location and velocity)

        self.my_filter.F = np.array([[1.,1.],
                        [0.,1.]])    # state transition matrix

        self.my_filter.H = np.array([[1.,0.]])    # Measurement function
        self.my_filter.P *= 1000                 # covariance matrix
        self.my_filter.R = 0.00001                      # state uncertainty
        

    #TODO abstract to update_measurement
    def update(self, dt):
        self.my_filter.Q = Q_discrete_white_noise(2, dt, .1) # process uncertainty
        sound_samples = [self.receiver.read_phone_mic(), self.receiver.read_pc_mic()]
        speeds = np.array([doppler_analyzer.extract_speeds_from(sound_samples[0 if i != 2 else 1], 1) for i, doppler_analyzer in enumerate(self.doppler_analyzers)])
        #print(speeds[2])
        # self.my_filter.predict()
        # self.my_filter.update(speeds[0]*dt)

        self.position.move_by(-speeds*dt)
        #self.speakers_distance = self.speaker_distance_finder.update(dt, speeds * dt)
        # for i, _ in enumerate(self.speakers):
        #     plotter.add_sample(f'predicted_x_position_{i}', self.get_distance()[i])
        # if self.two_dimensions:
        #     plotter.add_sample('predicted_y_position', self.get_distance()[1])


    def __del__(self):
        if pygame.mixer.get_init() is not None:
            pygame.mixer.stop()


class KalmanFiltery(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

class OfflinePredictor(Predictor):
    def __init__(self, config, plotter):
        self.config = config['config']
        self.options = config['options']
        Positioner.__init__(self, self.config, plotter)
        self.name = "predictor"
                
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.speakers = [Speaker(speaker_config) for speaker_config in self.config['speakers']]
        
        self.doppler_analyzers = [DopplerAnalyzer(speaker.get_config().get_frequencies(), plotter, config) for speaker in self.speakers]

        self.sound_samples = np.array([x for x in config['audio_samples']])
        self.cur_sound_samples = 0

        if 'kalman_filter' in self.options:
            dt = 1.0/60
            F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
            self.H = np.array([1, 0, 0]).reshape(1, 3)
            Q = np.array([[0.00001, 0.00001, 0.0], [0.00001, 0.00001, 0.0], [0.0, 0.0, 0.0]])
            R = np.array([0.00001]).reshape(1, 1)

            self.kf = KalmanFilter(F = F, H = self.H, Q = Q, R = R)

        self.speaker_distance_finder = SpeakerDistanceFinder(plotter)
        self.my_filter = KalmanFilter(dim_x=2, dim_z=1)
        self.my_filter.x = np.array([[0.],
                [0.]])       # initial state (location and velocity)

        self.my_filter.F = np.array([[1.,1.],
                        [0.,1.]])    # state transition matrix

        self.my_filter.H = np.array([[1.,0.]])    # Measurement function
        self.my_filter.P *= 1000                 # covariance matrix
        self.my_filter.R = 0.00001                      # state uncertainty
        
        
        
    def update(self, dt):
        self.my_filter.Q = Q_discrete_white_noise(2, dt, 0.00001) # process uncertainty

        sound_samples = self.sound_samples[self.cur_sound_samples]
        self.cur_sound_samples += 1
        x, y = self.position.get_position()
        xR, yR = self.position.get_other_position()
        cos = xR/np.sqrt(xR * xR + yR * yR)
        sin = yR/np.sqrt(xR * xR + yR * yR)
        cosL = (x/np.sqrt(x * x + y * y))
        sinL = (y/np.sqrt(x * x + y * y))
        cosines = (cosL * 0.70710678 + sinL * 0.70710678, cos * 0.70710678 + sin * 0.70710678)
        speeds = np.array([doppler_analyzer.extract_speeds_from(sound_samples, 1) for i, doppler_analyzer in enumerate(self.doppler_analyzers)])
        #self.speakers_distance = self.speaker_distance_finder.update(dt, speeds)
        
        self.my_filter.predict()
        self.my_filter.update(speeds[0]*dt)

        self.position.move_by(-self.my_filter.x[0])

        # if 'kalman_filter' in self.options:
        #     self.kf.F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        #     w = np.dot(self.H,  self.kf.predict())[0]
        #     print(w)
        #     self.plotter.add_sample('kalman_filter_x', w[0])
        #     if len(w) > 1:        
        #         self.plotter.add_sample('kalman_filter_y', w[1])
        #     else: 
        #         self.plotter.add_sample('kalman_filter_y', 0)
        #     self.kf.update(self.position.get_position())