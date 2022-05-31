import numpy as np
import sounddevice as sd
import threading

class SpeakerConfig:
    CHANNEL_ID = 1
    DEVICES = [4, 5]

    def __init__(self, config):
        self.name = config['name']
        self.frequencies = config['frequencies']
        if self.name == 'phone': return
        if len(SpeakerConfig.DEVICES) == 0:
            self.device = -1
            raise Exception("Not enough speakers available")
        self.device = SpeakerConfig.DEVICES[0]
        self.channel = SpeakerConfig.CHANNEL_ID
        SpeakerConfig.CHANNEL_ID += 1                   #0 if self.name == "left" else 1 if self.name == "right" else -1
        if SpeakerConfig.CHANNEL_ID % 3 == 0:
            SpeakerConfig.CHANNEL_ID = 1
            SpeakerConfig.DEVICES.pop(0)

    def get_channel(self):
        return self.channel

    def get_device(self):
        return self.device

    def get_name(self):
        return self.name

    def get_frequencies(self):
        return self.frequencies

class Speaker:
    def __init__(self, config):
        self.config = SpeakerConfig(config)

    # def play_sound(self, audio_samples, channels):        
    #     # if self.config.get_channel() == -1:
    #     #     return
    #     #audio_samples = self.get_audio_samples_of_frequencies(self.config.get_frequencies())
        
    #     def play():
    #         sd.play(audio_samples, 44100, mapping=[self.config.get_channel()], device=self.config.get_device(), loop=True)
    #         print(f"Speaker {self.config.get_name()} playing on channel {self.config.get_channel()} of device {self.config.get_device()}: {self.config.get_frequencies()}")#{sd.query_devices()[self.config.get_device()]}, channel {self.config.get_channel()}")
    #     thread = threading.Thread(target=play)
    #     thread.start()
        
    #     # sound = pygame.mixer.Sound(audio_samples)
    #     # channel = pygame.mixer.Channel(self.config.get_channel())
    #     # channel.play(sound, -1)
    #     # if self.config.get_channel() == 0:
    #     #     channel.set_volume(0.5, 0)
    #     # else: 
    #     #     channel.set_volume(0, 0.5)

    def get_config(self):
        return self.config

class PhoneSpeaker(Speaker):
    def __init__(self, config):
        self.config = SpeakerConfig(config)

    def play_sound(self):        
        pass

class SpeakersOrchestrator:
    def __init__(self, config) -> None:
        self.speakers = [PhoneSpeaker(speaker_config) if speaker_config['name'] == 'phone' else Speaker(speaker_config) for speaker_config in config['speakers']]
        speakers = [speaker for speaker in self.speakers if not isinstance(speaker, PhoneSpeaker)] #Discrd
        all_device_params = [[]]
        for speaker in speakers:
            config = speaker.get_config()
            curr_params = (config.get_device(), speaker)#, config.get_channel(), speaker.get_audio_samples_of_frequencies(config.get_frequencies()))
            
            if len(all_device_params[-1]) > 0 and curr_params[0] != all_device_params[-1][-1][0]: # If it is a different device
                all_device_params.append([curr_params])
            else:
                all_device_params[-1].append(curr_params)
        print(all_device_params)
        
        # For each unique device
        for device_params in all_device_params:
            if len(device_params) > 2:
                raise Exception("Invalid length")
            l_speaker_config = device_params[0][1].get_config()
            r_speaker_config = device_params[1][1].get_config()
            self.play_sound(device_params[0][0], l_speaker_config.get_frequencies(), r_speaker_config.get_frequencies())

    def play_sound(self, device, lFreqs, rFreqs):    
        print(f"Playing on device {device}. L: {lFreqs} R: {rFreqs}")

        if device == 5:
            return
        audio_samples = np.c_[self.get_audio_samples_of_frequencies(lFreqs), self.get_audio_samples_of_frequencies(rFreqs)]    
        
        def play():
            sd.play(audio_samples, 44100, mapping=[1, 2], device=device, loop=True)
            while True:
                pass           

        thread = threading.Thread(target=play)
        thread.start()
        
        # sound = pygame.mixer.Sound(audio_samples)
        # channel = pygame.mixer.Channel(self.config.get_channel())
        # channel.play(sound, -1)
        # if self.config.get_channel() == 0:
        #     channel.set_volume(0.5, 0)
        # else: 
        #     channel.set_volume(0, 0.5)
            
    def get_audio_samples_of_frequencies(self, frequencies):
        # [fStart, fEnd]
        sampleRate = 44100
        current_frequency = frequencies[0]
        samples = np.array([4096 * np.sin(2.0 * np.pi * current_frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
        for frequency in frequencies[1:-1]:
            samples -= np.array([4096 * np.sin(2.0 * np.pi * frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
        # Add last (instead of substract) samples in order to not overflow int16 range
        current_frequency = frequencies[-1]
        samples += np.array([4096 * np.sin(2.0 * np.pi * current_frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
        #final_samples = np.c_[samples, samples] # Make stereo samples (Sound() expected format)
        return samples

    def get_speakers(self):
        return self.speakers