import socket
from scipy import signal
import numpy as np
import pygame as pg
from scipy.io.wavfile import write
import time

pg.mixer.init(frequency=44100, size=-16,channels=2, buffer=4096)

def playSound2(f, speaker='L'):
    sound = pg.mixer.Sound(str(f) + 'k.wav')
    channel = 0 if speaker == 'L' else 1
    pgChannel = pg.mixer.Channel(channel)
    pgChannel.play(sound)
    if channel == 0:
        pgChannel.set_volume(0.5, 0.0)
    else: 
        pgChannel.set_volume(0.0, 0.5)

def playSound(fStart, fEnd, fHop, speaker='L'):
    sampleRate = 44100
    f = fStart
    finalSamples = np.array([4096 * np.sin(2.0 * np.pi * f * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
    sumar = True
    f += fHop
    while f != fEnd:
        if sumar: 
            finalSamples += np.array([4096 * np.sin(2.0 * np.pi * f * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
        else: 
            finalSamples -= np.array([4096 * np.sin(2.0 * np.pi * f * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
        sumar = False
        f += fHop
    arr2 = np.c_[finalSamples, finalSamples]
    sound = pg.mixer.Sound(arr2)
    channel = 0 if speaker == 'L' else 1
    pgChannel = pg.mixer.Channel(channel)
    pgChannel.play(sound, -1)
    if channel == 0:
        pgChannel.set_volume(0.45, 0.0)
    else: 
        pgChannel.set_volume(0.0, 0.45)

def getLSDoppler(Sxx):
    dopplerLS = np.array([np.argmax(Sxx[x-100:x+100]) - 100 for x in range(18000, 20000, 200)])
    indicesOfBest = np.abs(dopplerLS).argsort()[:5][::-1]
    bestDopplers = dopplerLS[indicesOfBest]
    bestFreqs = indicesOfBest * 200 + 18000
    vLS = [((bestDopplers[i]/x)*344.74 * 100) for i, x in enumerate(bestFreqs)]
    return np.mean(vLS)

def getRSDoppler(Sxx):
    dopplerRS = np.array([np.argmax(Sxx[x-100:x+100]) - 100 for x in range(20000, 22000, 200)])
    indicesOfBest = np.abs(dopplerRS).argsort()[:5][::-1]
    bestDopplers = dopplerRS[indicesOfBest]
    bestFreqs = indicesOfBest * 200 + 20000
    vLS = [((bestDopplers[i]/x)*344.74 * 100) for i, x in enumerate(bestFreqs)]
    return np.mean(vLS)

#Left speaker tones
playSound(18000, 20000, 200, 'L')
#playSound(20000, 22000, 200, 'R')

UDP_IP = "192.168.1.35"
UDP_PORT = 5555
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))


dt = 0
dL = 30 # Initial distance from left speaker to phone in cm
vL = 0
vR = 0
dR = 20 # Initial distance from right speaker to phone in cm
while True:
    start = time.time()
    data = sock.recv(2048) 
    length = int.from_bytes(data[0:4], "big")
    if length == 1796:        
        int_values = [x for x in data[4:length]]         
        _, _, Sxx = signal.spectrogram(np.array(int_values), fs=44100, nfft=44100, nperseg=1792, mode='magnitude')
        dopplerLS = getLSDoppler(Sxx)
        #dopplerRS = getRSDoppler(Sxx)
        dL = dL - dopplerLS*dt
        #dR = dR - dopplerRS*dt
        print(dL)#TODO NOW debugguear si 1hz doppler = 2cm/s en f correspondiente
        #print("DistanceLS: ", dL)
        #print("DistanceRS: ", dR)
        
        #print("DopplerLS: ", dopplerLS)
        # ponderar best por Sxx (desindad spectral de potencia)
        
        #print("indices of best", indicesOfBest)
        
        #print("best dopplers: ", bestDopplers)
        
        #print("best freqs: ", bestFreqs)
        
        #dopplerRS = [np.argmax(Sxx[x-100:x+100]) - 100 for x in range(20000, 22000, 200)]
        
        #print("DopplerRS: ", dopplerRS)

        
        
        #if np.abs(np.mean(vLS)) > 3:
        
        #vRS = (dopplerRS/21000)*346.6 * 100
        
        #print("vLS: ", vLS)
        #print("vRS: ", vRS)
    else:
        print("Invalid UDP packet.")
    dt = time.time() - start