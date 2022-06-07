import numpy as np
import pygame
from pygame import _sdl2
import multiprocess
import threading

class SpeakerConfig:
    def __init__(self, config):
        self.name = config['name']
        self.frequencies = config['frequencies']
        if "type" in config and config["type"] == 'virtual': return
        devices = SpeakerOrchestrator.get_audio_devices_names()
        self.device = devices[config['device']]
        self.channel = config['channel']

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
        def play(device_name, channel_number, frequencies, audio_samples):    
            pygame.mixer.pre_init(devicename=device_name)
            pygame.mixer.init()                    
            sound = pygame.mixer.Sound(audio_samples)
            mixer_channel = pygame.mixer.Channel(channel_number)
            
            mixer_channel.play(sound, -1)
            if channel_number == 0:
                mixer_channel.set_volume(0.5, 0)
            else:
                mixer_channel.set_volume(0, 0.5)
            print(f"Playing frequencies {frequencies} on device's {device_name} channel {channel_number}")
            
            import time
            while True:
                time.sleep(100000)

        audio_samples = self.generate_audio_samples(self.config.get_frequencies())
        
        def start_proc():
            self.sound_process = multiprocess.Process(target=play, args=(self.config.get_device(), self.config.get_channel(), self.config.get_frequencies(), audio_samples))
            self.sound_process.start()

        # We need to create one process for each channel to work with multiple devices
        threading.Thread(target=start_proc).start()

    def stop_sound(self):
        self.sound_process.terminate()

    def generate_audio_samples(self, frequencies):
        SAMPLE_RATE = 44100
        AMPLITUDE = 1024
        DURATION = 1.0
        LENGTH = int(SAMPLE_RATE * DURATION)

        audio_samples = np.fromfunction(lambda x: AMPLITUDE * np.sin(2.0 * np.pi * frequencies[0] * x / SAMPLE_RATE), (LENGTH,)).astype(np.int16)
        for frequency in frequencies[1:-1]:
            audio_samples -= np.fromfunction(lambda x: AMPLITUDE * np.sin(2.0 * np.pi * frequency * x / SAMPLE_RATE), (LENGTH,)).astype(np.int16)
        # Add (instead of substract) latest samples in order to not overflow int16 range
        audio_samples += np.fromfunction(lambda x: AMPLITUDE * np.sin(2.0 * np.pi * frequencies[-1] * x / SAMPLE_RATE), (LENGTH,)).astype(np.int16)

        stereo_samples = np.c_[audio_samples, audio_samples] # Make samples stereo (pygame.Sound() expected format)
        return stereo_samples

    def get_config(self):
        return self.config

class VirtualSpeaker(Speaker):
    def play_sound(self):        
        pass


class SpeakerOrchestrator:
    def __init__(self, config):
        self.speakers = [VirtualSpeaker(speaker_config) if 'type' in speaker_config and speaker_config['type'] == 'virtual' else Speaker(speaker_config) for speaker_config in config['speakers']]
        
    def play_sound(self):        
        for speaker in self.speakers:
            speaker.play_sound()

    def get_speakers(self):
        return self.speakers

    def stop_sound(self):
        for speaker in self.speakers:
            speaker.stop_sound()

    @staticmethod
    def get_audio_devices_names():
        pygame.init()
        devices_names = pygame._sdl2.get_audio_device_names()
        pygame.quit()
        return devices_names