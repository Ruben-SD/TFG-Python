import pygame
from positioner import Positioner
from speaker import Speaker
from receiver import Receiver
import plotter
from doppleranalyzer import DopplerAnalyzer
from plotter import *
from scipy import signal

class Predictor(Positioner):
    def __init__(self, config):
        super().__init__(config)
        self.name = "predictor"
        self.sound_samples = []
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.speakers = [Speaker(speaker_config) for speaker_config in config['speakers']]
        # for speaker in self.speakers:
        #    self.sound_samples = speaker.play_sound()
        self.sound_samples = np.array(self.speakers[0].play_sound()[:44100*50])

        # for x in range(100):
        #     self.sound_samples = self.sound_samples + self.sound_samples
        self.sound_samples = (self.sound_samples - min(self.sound_samples)) / (max(self.sound_samples) - min(self.sound_samples))
        self.receiver = Receiver()
        self.all_samples = []
        self.time = None
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
        if self.time is None:
            self.time = 0
        import wave
        import struct
        from numpy import array, concatenate, argmax
        from numpy import abs as nabs
        from scipy.signal import fftconvolve
        from matplotlib.pyplot import plot, show
        from math import log

        def crossco(wav):
            """Returns cross correlation function of the left and right audio. It
            uses a convolution of left with the right reversed which is the
            equivalent of a cross-correlation.
            """
            cor = nabs(fftconvolve(wav[0],wav[1][::-1]))
            return cor

        self.time += dt
        sound_samples = self.receiver.retrieve_sound_samples()
    
        self.all_samples = list(self.all_samples) + list(sound_samples)
        
        all_norm_samples = (self.all_samples - min(self.all_samples)) / (max(self.all_samples) - min(self.all_samples)) 
        
        if len(all_norm_samples) > 44100 * 10:
            cor = np.argmax(signal.correlate(self.sound_samples, all_norm_samples))
            print((1-(cor/(44100 *50)))*344.44*100)
        # plt.plot(np.arange(len(self.all_samples)), np.abs(self.all_samples))
        # plt.show()
        
        speeds = DopplerAnalyzer.extract_speeds_from(sound_samples, [speaker.get_config().get_frequencies() for speaker in self.speakers])
        self.position.move_by(-np.array(speeds) * dt)
        # for i, _ in enumerate(self.speakers):
        #     plotter.add_sample(f'predicted_x_position_{i}', self.get_distance()[i])
        # if self.two_dimensions:
        #     plotter.add_sample('predicted_y_position', self.get_distance()[1])


    def __del__(self):
        if pygame.mixer.get_init() is not None:
            pygame.mixer.stop()