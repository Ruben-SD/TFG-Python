
import socket
import numpy as np
import pygame as pg

import time


def play_sound():        
    audio_samples = get_audio_samples_of_frequencies([18000, 18200, 18400, 18600, 18800, 19000])
    sound = pg.mixer.Sound(audio_samples)
    channel = pg.mixer.Channel(0)
    channel.play(sound, -1)
    channel.set_volume(0.5, 0)

def get_audio_samples_of_frequencies(frequencies):
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



pg.mixer.init(frequency=44100, size=-16,channels=2, buffer=4096)


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
UDP_IP = s.getsockname()[0]
UDP_PORT = 5555
s.close()
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))
print("Listening on: ", UDP_IP, ":", UDP_PORT)


init = time.time()
timestamp = time.strftime("%d-%m-%Y-%H:%M:%S")
i = 0
arr2 = np.c_[np.ones(44100) * 40960, np.ones(44100) * 40960] # Make stereo samples
sound = pg.mixer.Sound(arr2)

pgChannel = pg.mixer.Channel(0)

a  = time.time()
played = False

while True:
    if time.time() - a > 3 and not played:
        print("Played")
        played = True
        play_sound()
        start = time.perf_counter_ns()    
    
    data = sock.recv(2048)     
    x = [0, 0, 0, data[0]]
    s = [0, 0, 0, data[1]]
    
    length = int.from_bytes(x, "big")
    sound2 = int.from_bytes(s, "big")
    
    if sound2 != 128:        
        dt = time.perf_counter_ns() - start
        print(dt/10e9)
        print((344.44*100) * (dt/10e9))
        break
    
    if (length != i and i != 0 ):
        break
    if length == i:
        if i == 255:
            i = -1
        i += 1