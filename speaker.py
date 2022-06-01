import numpy as np
import pygame
import pygame._sdl2 as sdl2
from multiprocess import Process

class SpeakerConfig:
    CHANNEL_ID = 0
    DEVICES = [0, 1]

    def __init__(self, config):
        self.name = config['name']
        self.frequencies = config['frequencies']
        if self.name == 'phone': return
        if len(SpeakerConfig.DEVICES) == 0:
            raise Exception("Not enough speakers available")
        self.device = SpeakerConfig.DEVICES[0]
        self.channel = SpeakerConfig.CHANNEL_ID
        SpeakerConfig.CHANNEL_ID += 1                   #0 if self.name == "left" else 1 if self.name == "right" else -1
        if SpeakerConfig.CHANNEL_ID % 2 == 0:
            SpeakerConfig.CHANNEL_ID = 0
            SpeakerConfig.DEVICES.pop(0)

    def get_channel(self):
        return self.channel

    def get_device(self):
        return self.device

    def get_frequencies(self):
        return self.frequencies

class Speaker:
    def __init__(self, config):
        self.config = SpeakerConfig(config)

    def play_sound(self):        
        def play(device, channel, audio_samples):
            #print(f"Playing on device {device} {names[device]} channel {channel}")
            #pygame.quit()

            pygame.mixer.pre_init(devicename=device)
            pygame.mixer.init()                    
            if channel == 0:
                sound = pygame.mixer.Sound(audio_samples)
            else:
                sound = pygame.mixer.Sound(audio_samples)
            mixer_channel = pygame.mixer.Channel(channel)
            
            mixer_channel.play(sound, -1)
            if channel == 0:
                mixer_channel.set_volume(0.5, 0)
            else:
                mixer_channel.set_volume(0, 0.5)
            import time
            while True:
                time.sleep(1)

        audio_samples = self.get_audio_samples_of_frequencies(self.config.get_frequencies())
        print(self.config.get_device(), self.config.get_channel(), self.config.get_frequencies())
        pygame.init()
        names = sdl2.get_audio_device_names()
        print(names, self.config.get_device())
        p = Process(target=play, args=(names[self.config.get_device()], self.config.get_channel(), audio_samples))
        p.start()
        #crear process y pasarle args

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

class PhoneSpeaker(Speaker):
    def __init__(self, config):
        self.config = SpeakerConfig(config)

    def play_sound(self):        
        pass

    