
from os import times
from random import sample
import socket
import numpy as np
import pygame as pg

from scipy import signal

import time

start = None

def get_audio_samples_of_frequencies(frequencies):
    # [fStart, fEnd]
    sampleRate = 96000
    current_frequency = frequencies[0]
    samples = [1024 * np.sin(2.0 * np.pi * current_frequency * x / sampleRate) for x in range(sampleRate)]
    samples = samples + [0 for x in range(sampleRate)]
    for x in range(5):
        samples = samples + samples
    samples = np.array(samples).astype(np.int16)
    final_samples = np.c_[samples, samples] # Make stereo samples (Sound() expected format)
    return final_samples


def play_sound():        
    audio_samples = get_audio_samples_of_frequencies([500, 18200, 18400, 18600, 18800, 19000, 500, 1000, 1600, 2500])
    sound = pg.mixer.Sound(audio_samples)
    channel = pg.mixer.Channel(0)
    from  scipy.io import wavfile 
    wavfile.write('sound500.wav', 96000, audio_samples)
    global start
    
    
    channel.set_volume(1, 0)
    start = time.time_ns()/1000
    channel.play(sound)
    print("Done")

# pg.mixer.init(frequency=44100, size=-16,channels=2, buffer=512)

# play_sound()


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




# import rtmixer
# import sounddevice as sd
# import soundfile as sf
# starttime = None
# def thisf():
#     filename = "sound.wav"
#     playback_blocksize = None
#     latency = None
#     reading_blocksize = 1024  # (reading_blocksize * rb_size) has to be power of 2
#     rb_size = 16  # Number of blocks

#     with sf.SoundFile(filename) as f:
#         with rtmixer.Mixer(channels=f.channels,
#                         blocksize=playback_blocksize,
#                         samplerate=f.samplerate, latency=latency) as m:
#             elementsize = f.channels * m.samplesize
#             rb = rtmixer.RingBuffer(elementsize, reading_blocksize * rb_size)
#             # Pre-fill ringbuffer:
#             _, buf, _ = rb.get_write_buffers(reading_blocksize * rb_size)
#             written = f.buffer_read_into(buf, dtype='float32')
#             rb.advance_write_index(written)
#             action = m.play_ringbuffer(rb)
#             global starttime
#             #starttime = time.time_ns()/10e9
#             while True:
#                 while rb.write_available < reading_blocksize:
#                     if action not in m.actions:
#                         break
#                     sd.sleep(int(1000 * reading_blocksize / f.samplerate))
#                 if action not in m.actions:
#                     break
#                 size, buf1, buf2 = rb.get_write_buffers(reading_blocksize)
#                 assert not buf2
#                 written = f.buffer_read_into(buf1, dtype='float32')
#                 rb.advance_write_index(written)
#                 if written < size:
#                     break
#             m.wait(action)
#             if action.done_frames != f.frames:
#                 RuntimeError('Something went wrong, not all frames were played')
#             if action.stats.output_underflows:
#                 print('output underflows:', action.stats.output_underflows)



# sound = pg.mixer.Sound(arr2)

# pgChannel = pg.mixer.Channel(0)

a  = time.time()
played = False
last = 0
start = 0
wait = 0
from threading import Thread
# #thread.start()
beepi = 0
other = 0
last = -1

greater = -1
import keyboard
starttime = -1
samples = []
while not keyboard.is_pressed('q'):
    data = sock.recv(2048)  
    import array
    arr = array.array('f', data[8:])
    samples = samples + arr.tolist()
    
    if len(samples) >= 1792:            
        _, _, Sxx = signal.spectrogram(np.array(samples), fs=48000, nfft=48000, nperseg=len(samples), mode='magnitude')
        dopplerRS = np.argmax(Sxx[450:550])
        if dopplerRS == 50 and time.time() > wait:
            wait = time.time() + 1.2
            timestamp = int.from_bytes(data[:8], "little")
            dt = (timestamp/10e6 - (starttime/10e6))
            starttime = timestamp
            print(344.44 * 100 * dt, dt)
        samples = []