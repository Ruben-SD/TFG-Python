import pygame
from positioner import Positioner
from speaker import Speaker
from receiver import Receiver
import plotter
from doppleranalyzer import DopplerAnalyzer
from plotter import *

class SpeakerDistanceFinder:
    def __init__(self):
        self.speeds = []
        self.positive = None

    def update(self, speeds):
        print(speeds)
        if np.abs(speeds)[0] < 20 and np.abs(speeds)[1] < 20:
            return
        self.speeds.append(speeds)
        if self.positive is None:
            self.positive = [speeds[0] > 0, speeds[1] > 0]
        else:
            if (speeds[0] < 0 and self.positive[0]) or (speeds[0] > 0 and not self.positive[0]):
                print("LEFT\n")
                self.positive[0] = not self.positive[0]
                #self.speeds = []
            if (speeds[1] < 0 and self.positive[1]) or (speeds[1] > 0 and not self.positive[1]):
                self.positive[1] = not self.positive[1]
                print("RIGHT\n")
                #self.speeds = []
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
        self.position.move_by(-np.array(speeds) * dt)
        self.speaker_distance_finder.update(speeds)
        #speaker_distance = self.speaker_distance_finder.update(speeds)
        #print(speaker_distance)


    def __del__(self):
        if pygame.mixer.get_init() is not None:
            pygame.mixer.stop()