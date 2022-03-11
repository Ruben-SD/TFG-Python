from math import dist
import pygame
from positioner import Positioner
from speaker import Speaker
from receiver import Receiver
import plotter
from doppleranalyzer import DopplerAnalyzer
from plotter import *

class SpeakerDistanceFinder:
    def __init__(self):
        self.distances = np.zeros(2)
        self.time = 0
        self.is_on_left = True
        self.delay = False
        self.all_distances = []

    def update(self, dt, displacements):
        self.time += dt
        self.distances += np.abs(displacements)

        if self.time > 0.5 and self.is_on_left and displacements[0] > 0 and self.distances[0] > 5:
            self.time = 0            
            distance = self.distances[0]
            self.all_distances.append(distance)
            print(distance)
            self.distances = displacements
            self.is_on_left = False
        if self.time > 0.5 and not self.is_on_left and displacements[1] > 0 and self.distances[1] > 5:
            self.time = 0
            distance = self.distances[1]
            self.all_distances.append(distance)
            print(distance)
            self.distances = displacements
            self.is_on_left = True 
        print("Distance: ", np.mean(self.all_distances))

class Predictor(Positioner):
    def __init__(self, config):
        super().__init__(config)
        self.name = "predictor"
        
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.speakers = [Speaker(speaker_config) for speaker_config in config['speakers']]
        for speaker in self.speakers:
            speaker.play_sound()
        
        self.receiver = Receiver()
        self.speaker_distance_finder = SpeakerDistanceFinder()

    def update(self, dt):
        sound_samples = self.receiver.retrieve_sound_samples()
        speeds = DopplerAnalyzer.extract_speeds_from(sound_samples, [speaker.get_config().get_frequencies() for speaker in self.speakers])
        displacements = -np.array(speeds) * dt
        self.position.move_by(displacements)
        self.speakers_distance = self.speaker_distance_finder.update(dt, displacements)


    def __del__(self):
        if pygame.mixer.get_init() is not None:
            pygame.mixer.stop()