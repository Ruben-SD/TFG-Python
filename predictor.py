import pygame
from positioner import Positioner
from speaker import Speaker
from receiver import Receiver
import plotter
from doppleranalyzer import DopplerAnalyzer
from plotter import *

class Predictor(Positioner):
    def __init__(self, config):
        super().__init__(config)
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.speakers = [Speaker(speaker_config) for speaker_config in config['speakers']]
        for speaker in self.speakers:
            speaker.play_sound()
        self.receiver = Receiver()
        plotter.add_data('predicted_x_position', [], plot=True)

    #TODO abstract to update_measurement
    def update_position(self, dt):
        sound_samples = self.receiver.read_packet()
        speeds = DopplerAnalyzer.get_speeds_from(sound_samples, [speaker.get_config().get_frequencies() for speaker in self.speakers])#, 
        # asbtract to measurement
        #self.position.add_speed(vx * dt, vy * dt)
        #todo   
        self.position = (self.position[0] + speeds[0] * dt, 0)
        plotter.add_sample('predicted_x_position', self.position[0])
        return self.position

    def __del__(self):
        if pygame.mixer.get_init() is not None:
            pygame.mixer.stop()