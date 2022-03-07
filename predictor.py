import pygame
from positioner import Positioner
from speaker import Speaker
from receiver import Receiver
import plotter
from doppleranalyzer import DopplerAnalyzer
from plotter import *

class SpeakerDistanceFinder:
    def __init__(self):
        self.speeds_left = []
        self.speeds_right = []
        self.is_on_left = True
        self.delay = False

    def update(self, speeds):
        if np.all(np.abs(speeds) < 0.2):
            return

        self.speeds_left.append(speeds[0])
        self.speeds_right.append(speeds[1])

        if len(self.speeds_left) > 30 and self.is_on_left and speeds[0] >= 1:            
            distance = np.sum(self.speeds_left) # promediar con distances anteriores
            if np.abs(distance) > 4:
                self.delay = True
                self.start = time.time()
                print(f"In Left speaker, distance = {distance}")
                self.speeds_left = []
                self.speeds_right = []
                self.is_on_left = False
        if len(self.speeds_right) > 30 and not self.is_on_left and speeds[1] >= 1:
            distance = np.sum(self.speeds_right)
            if np.abs(distance) > 4:
                print(f"In Right speaker, distance = {distance}")
                self.delay = True
                self.start = time.time()
                self.speeds_left = []              
                self.speeds_right = [] 
                self.is_on_left = True 
            #print distance = integrate self.speeds from last interval change

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
        self.speakers_distance = self.speaker_distance_finder.update(displacements)


    def __del__(self):
        if pygame.mixer.get_init() is not None:
            pygame.mixer.stop()