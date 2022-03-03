import numpy as np
import pygame

class SpeakerConfig:
    def __init__(self, config):
        self.name = config['name']
        self.frequencies = config['frequencies']
        self.channel = 0 if self.name == "left" else 1 

    def get_channel(self):
        return self.channel

    def get_frequencies(self):
        return self.frequencies

class Speaker:
    def __init__(self, config):
        self.config = SpeakerConfig(config)

    def play_sound(self):        
        audio_samples = self.get_audio_samples_of_frequencies(self.config.get_frequencies())
        sound = pygame.mixer.Sound(audio_samples)
        channel = pygame.mixer.Channel(self.config.get_channel())
        channel.play(sound, -1)
        if self.config.get_channel() == 0:
            channel.set_volume(0.45, 0)
            print("a")
        else: 
            channel.set_volume(0, 0.45)
            print("b")

    def get_audio_samples_of_frequencies(self, frequencies):
        # [fStart, fEnd]
        sampleRate = 44100
        current_frequency = frequencies[0]
        samples = np.array([1024 * np.sin(2.0 * np.pi * current_frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
        for frequency in frequencies[1:-1]:
            samples -= np.array([1024 * np.sin(2.0 * np.pi * frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
        # Add last (instead of substract) samples in order to not overflow int16 range
        current_frequency = frequencies[-1]
        samples += np.array([1024 * np.sin(2.0 * np.pi * current_frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
        final_samples = np.c_[samples, samples] # Make stereo samples (Sound() expected format)
        return final_samples

    def get_config(self):
        return self.config