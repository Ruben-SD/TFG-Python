from pykalman import KalmanFilter
import pygame
from positioner import Positioner
from speaker import Speaker
from receiver import Receiver
from doppleranalyzer import DopplerAnalyzer
import numpy as np

class Predictor(Positioner):
    def __init__(self, config, plotter):
        super().__init__(config, plotter)
        self.name = "predictor"
        
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.speakers = [Speaker(speaker_config) for speaker_config in config['speakers']]
        for speaker in self.speakers:
            speaker.play_sound()
        
        self.receiver = Receiver()
        self.doppler_analyzers = [DopplerAnalyzer(speaker.get_config().get_frequencies(), plotter, config) for speaker in self.speakers]
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

    #TODO abstract to update_measurement
    def update(self, dt):
        sound_samples = self.receiver.retrieve_sound_samples()
        angles = self.position
        speeds = np.array([doppler_analyzer.extract_speeds_from(sound_samples) for doppler_analyzer in self.doppler_analyzers])
        self.position.move_by(-np.array(speeds) * dt)
        # for i, _ in enumerate(self.speakers):
        #     plotter.add_sample(f'predicted_x_position_{i}', self.get_distance()[i])
        # if self.two_dimensions:
        #     plotter.add_sample('predicted_y_position', self.get_distance()[1])


    def __del__(self):
        if pygame.mixer.get_init() is not None:
            pygame.mixer.stop()


class KalmanFilter(object):
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
        
        
    def update(self, dt):
        sound_samples = self.sound_samples[self.cur_sound_samples]
        self.cur_sound_samples += 1
        x, y = self.position.get_position()
        xR, yR = x + self.speakers_distance, y
        cosines = ((y/np.sqrt(x * x + y * y)), y/np.sqrt(xR * xR + yR * yR))
        speeds = np.array([doppler_analyzer.extract_speeds_from(sound_samples, cosines[i]) for i, doppler_analyzer in enumerate(self.doppler_analyzers)])
        self.position.move_by(-np.array(speeds) * dt)

        if 'kalman_filter' in self.options:
            self.kf.F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
            w = np.dot(self.H,  self.kf.predict())[0]
            print(w)
            self.plotter.add_sample('kalman_filter_x', w[0])
            if len(w) > 1:        
                self.plotter.add_sample('kalman_filter_y', w[1])
            else: 
                self.plotter.add_sample('kalman_filter_y', 0)
            self.kf.update(self.position.get_position())