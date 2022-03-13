
from os import times
import socket
import numpy as np
import pygame as pg

import time

start = None
def play_sound():        
    audio_samples = get_audio_samples_of_frequencies([18000, 18200, 18400, 18600, 18800, 19000, 500, 1000, 1600, 2500])
    sound = pg.mixer.Sound(audio_samples)
    channel = pg.mixer.Channel(0)
    
    global start
    
    
    channel.set_volume(1, 0)
    start = time.time_ns()/1000
    channel.play(sound, -1)

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



pg.mixer.init(frequency=44100, size=-16,channels=2, buffer=512)


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
UDP_IP = s.getsockname()[0]
UDP_PORT = 5555
s.close()
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))
print("Listening on: ", UDP_IP, ":", UDP_PORT)
pg.mixer.pre_init(44100, -16, 1, 512)
pg.init()

init = time.time()
timestamp = time.strftime("%d-%m-%Y-%H:%M:%S")
i = 0
arr2 = np.c_[np.ones(44100) * 40960, np.ones(44100) * 40960] # Make stereo samples



sound = pg.mixer.Sound(arr2)

pgChannel = pg.mixer.Channel(0)

a  = time.time()
played = False


#!/usr/bin/env python3
"""Play some random bleeps.

This example shows the feature of playing a buffer at a given absolute time.
Since all play_buffer() invocations are (deliberately) done at once, this puts
some strain on the "action queue".
The "qsize" has to be increased in order to handle this.

This example also shows that NumPy arrays can be used, as long as they are
C-contiguous and use the 'float32' data type.

"""


import numpy as np
import rtmixer
import sounddevice as sd

seed = 99

device = None
blocksize = 0
latency = 'low'
samplerate = 44100

bleeps = 300
qsize = 512  # Must be a power of 2

attack = 0.005
release = 0.1
pitch_min = 40
pitch_max = 80
duration_min = 0.2
duration_max = 0.6
amplitude_min = 0.05
amplitude_max = 0.15
start_min = 0
start_max = 10
channels = None

if duration_min < max(attack, release):
    raise ValueError('minimum duration is too short')

fade_in = np.linspace(0, 1, num=int(samplerate * attack))
fade_out = np.linspace(1, 0, num=int(samplerate * release))

r = np.random.RandomState(seed)

bleeplist = []

if channels is None:
    channels = sd.default.channels['output']
    if channels is None:
        channels = sd.query_devices(device, 'output')['max_output_channels']

for _ in range(bleeps):
    duration = r.uniform(duration_min, duration_max)
    amplitude = r.uniform(amplitude_min, amplitude_max)
    pitch = r.uniform(pitch_min, pitch_max)
    # Convert MIDI pitch (https://en.wikipedia.org/wiki/MIDI_Tuning_Standard)
    frequency = 2 ** ((pitch - 69) / 12) * 440
    t = np.arange(int(samplerate * duration)) / samplerate
    bleep = amplitude * np.sin(2 * np.pi * frequency * t, dtype='float32')
    bleep[:len(fade_in)] *= fade_in
    bleep[-len(fade_out):] *= fade_out

    # Note: Arrays must be 32-bit float and C contiguous!
    assert bleep.dtype == 'float32'
    assert bleep.flags.c_contiguous
    bleeplist.append(bleep)

actionlist = []


with rtmixer.Mixer(device=device, channels=channels, blocksize=blocksize,
                samplerate=samplerate, latency=latency, qsize=qsize) as m:
    while True:
        if time.time() - a > 3 and not played:
                start_time = m.time
                start = time.time_ns()/1000
                actionlist = [
                    m.play_buffer(bleep,
                                    channels=[r.randint(channels) + 1],
                                    start=start_time + r.uniform(start_min, start_max),
                                    allow_belated=False)
                    for bleep in bleeplist
                ]
                played = True

        data = sock.recv(2048)     
        x = [0, 0, 0, data[0]]
        s = [0, 0, 0, data[9]]

        timestamp = int.from_bytes(data[1:9], "big")
        times = time.time_ns()/1000

        length = int.from_bytes(x, "big")

        sound2 = int.from_bytes(s, "big")

        if sound2 != 128:        
            timestamp = int.from_bytes(data[1:9], "big")
            dt = (timestamp - start)/10e6 + 0.19093565
            print(dt)
            print((344.44*100) * dt)
            break

        if (length != i and i != 0):
            break
        if length == i:
            if i == 255:
                i = -1
            i += 1