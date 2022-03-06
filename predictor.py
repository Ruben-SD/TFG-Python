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
        self.name = "predictor"
        
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.speakers = [Speaker(speaker_config) for speaker_config in config['speakers']]
        for speaker in self.speakers:
            speaker.play_sound()
        
        self.receiver = Receiver()
        self.sound = []
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
        self.sound = self.sound + list(sound_samples)
        speeds = DopplerAnalyzer.extract_speeds_from(sound_samples, [speaker.get_config().get_frequencies() for speaker in self.speakers])
        self.position.move_by(-np.array(speeds) * dt)
        # for i, _ in enumerate(self.speakers):
        #     plotter.add_sample(f'predicted_x_position_{i}', self.get_distance()[i])
        # if self.two_dimensions:
        #     plotter.add_sample('predicted_y_position', self.get_distance()[1])


    def __del__(self):
        if pygame.mixer.get_init() is not None:
            pygame.mixer.stop()
        import pickle
        with open('sound', 'wb') as f:
           pickle.dump(self.sound, f)