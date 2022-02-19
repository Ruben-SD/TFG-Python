#from os import O_NONBLOCK
from pydoc import describe
from random import sample
import socket
from this import d
from scipy import signal
import numpy as np
import pygame as pg
from scipy.io.wavfile import write
import time

mergedDopplers = []
usedDopplers = []

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
    # [fStart, fEnd)
    sampleRate = 44100
    f = fStart
    finalSamples = np.array([4096 * np.sin(2.0 * np.pi * f * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
    sumar = True
    f += fHop
    while f != fEnd:
        if sumar: # Don't add always so as to not overflow int16 range
            finalSamples += np.array([4096 * np.sin(2.0 * np.pi * f * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
        else: 
            finalSamples -= np.array([4096 * np.sin(2.0 * np.pi * f * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
        sumar = False
        f += fHop
    # samples = []
    # for i in range(0, 44100, 100):
    #     samples = samples + [4096 * np.sin(2.0 * np.pi * 17800 * x / sampleRate) for x in range(i, i + 50)]
    #     samples = samples + [0 for x in range(i, i + 50)]
    #     if i > 44100: 
    #         break
    #     #finalSamples -= np.array([0 for x in range(i + 100, i + 200)]).astype(np.int16)
    # finalSamples -= np.array(samples).astype(np.int16)
    arr2 = np.c_[finalSamples, finalSamples] # Make stereo samples
    sound = pg.mixer.Sound(arr2)
    channel = 0 if speaker == 'L' else 1
    pgChannel = pg.mixer.Channel(channel)
    pgChannel.play(sound, -1)
    if channel == 0:
        pgChannel.set_volume(0.45, 0.0)
    else: 
        pgChannel.set_volume(0.0, 0.45)

def getLSDoppler(Sxx):
    dopplerLS = np.array([np.argmax(Sxx[x-100:x+100]) - 100 for x in range(18000, 20000, 200)]) # Get doppler desviation at each frequency
    dopplerLS = dopplerLS + .23424 #to avoid zeroing
    an_array=dopplerLS
    mean = np.mean(an_array)
    standard_deviation = np.std(an_array)
    distance_from_mean = abs(an_array - mean)
    max_deviations = 1.35
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    dopplerLS = an_array[not_outlier]
    dopplerLS = dopplerLS - .23424
    indicesOfBest = np.abs(dopplerLS).argsort()[:5][::-1] # Get indices of 5 greatest desviations (all)
    bestDopplers = dopplerLS[indicesOfBest] # Get 5 greatest desviations
    bestFreqs = indicesOfBest * 200 + 18000 # Compute doppler frequency for each best desviation
    vLS = [((bestDopplers[i]/x)*344.74 * 100) for i, x in enumerate(bestFreqs)] # Compute speed in cm/s for each doppler
    mergedDopplers.append(vLS)
    #print(np.mean(vLS))
    return np.mean(vLS) # Return mean of computed speeds

def getRSDoppler(Sxx):
    dopplerRS = np.array([np.argmax(Sxx[x-100:x+100]) - 100 for x in range(20000, 22000, 200)])
    indicesOfBest = np.abs(dopplerRS).argsort()[:5][::-1]
    bestDopplers = dopplerRS[indicesOfBest]
    bestFreqs = indicesOfBest * 200 + 20000
    vLS = [((bestDopplers[i]/x)*344.74 * 100) for i, x in enumerate(bestFreqs)]
    return np.mean(vLS)

# #Left speaker tones
# playSound(18000, 20000, 200, 'L')
# #playSound(20000, 22000, 200, 'R')

# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# s.connect(("8.8.8.8", 80))
# UDP_IP = s.getsockname()[0]
# UDP_PORT = 5555
# s.close()
# sock = socket.socket(socket.AF_INET, # Internet
#                      socket.SOCK_DGRAM) # UDP
# sock.bind((UDP_IP, UDP_PORT))
# print("Listening on: ", UDP_IP, ":", UDP_PORT)

# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
# f = KalmanFilter (dim_x=2, dim_z=1)

# f.x = np.array([[20.],    # position
#                 [0.]])   # velocity

# f.H = np.array([[0.,1.]]) # measurement function?

# f.P *= 0.                #modify (too large?)

def getBlackSquareContour(contours, gray):
    maxArea = -1
    index = -1
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        mean = cv2.mean(gray[y:y+h, x:x+w])[0]
        area = cv2.contourArea(contour)

        if area > maxArea and mean < 80 and 150 <= y <= 300:
            index = i
            maxArea = area
    if index == -1:
        raise ValueError("Invalid contour index")
    return contours[index]

import cv2

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid.set(cv2.CAP_PROP_AUTOFOCUS, 0) 

# objpoints = np.load('objpoints.csv.npy')
# imgpoints = np.load('imgpoints.csv.npy')


# print(objpoints)
# print(imgpoints)

# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (640, 480), None, None)

# ret, img = vid.read()
# h,  w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

lastPos = 1
pos = 80

realPosData = []
posData = []
timeData = []

startTime = time.time()
cmPerPx = -1


#Left speaker tones
playSound(18000, 20000, 200, 'L')
#playSound(20000, 22000, 200, 'R')

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
UDP_IP = s.getsockname()[0]
UDP_PORT = 5555
s.close()
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))
print("Listening on: ", UDP_IP, ":", UDP_PORT)

dt = 0
dL = pos # Initial distance from left speaker to phone in cm
vL = 0
vR = 0
dR = 80 # Initial distance from right speaker to phone in cm
init = time.time()
timestamp = time.strftime("%d-%m-%Y-%H:%M:%S")
i = 0
a = -1
while True:
    start = time.perf_counter_ns()
    data = sock.recv(2048)     
    x = [0, 0, 0, data[0]]
    
    length = int.from_bytes(x, "big")
    print(length, i)
    if (length != i and i != 0 ):
        break
    if length == i:
        if i == 255:
            i = -1

        
        # if time.time() - 0.5 < init:
        #     continue
        # int_values = [x for x in data[4:length]]         
        # _, _, Sxx = signal.spectrogram(np.array(int_values), fs=44100, nfft=44100, nperseg=1792, mode='magnitude')
        
        # dopplerRS = getRSDoppler(Sxx)
        # dR = dR - dopplerRS * dt
        # dopplerLS = getLSDoppler(Sxx)
        # usedDopplers.append(dopplerLS)
        # dL = dL + dopplerLS*dt
        
        # #dR = dR - dopplerRS*dt
        # #TODO NOW debugguear si 1hz doppler = 2cm/s en f correspondiente

        # start2 = time.time()
        # ret, frame = vid.read()
        
        # # dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # # # crop the image
        # # x, y, w, h = roi
        # # dst = dst[y:y+h, x:x+w]

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        
        # ret, binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
        # eroded = binary.copy()
        # cv2.erode(binary, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1]), eroded)
        
        # contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # blackSquareContour = getBlackSquareContour(contours, gray)
        # cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)    
        # x, y, w, h = cv2.boundingRect(blackSquareContour)
        # x = int(x + w/2)
        # if cmPerPx == -1:
        #     cmPerPx = 17.3/w
        #     camPos = x
            
        # realPos = (x-camPos)*cmPerPx + pos
        # #print(dL, dR, realPos, x)
        

        # #print(x, y, -(x-582)*cmPerPx + pos)
        # cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        # cv2.imshow("img", frame)
        # cv2.waitKey(1)
        # dt = time.time() - start2
        # posData.append(dL)
        # realPosData.append(realPos)
        # timeData.append(time.time() - startTime)
        
    
        # speed = ((x - lastPos)*cmPerPx)/dt
        # #print("Speed = ", speed, " cm/s")
        # lastPos = x
        # import keyboard
        # if keyboard.is_pressed('q'):
        #     break
        i += 1
        # print("FPS: ", (time.perf_counter_ns() - start)/1000)
        # dt = time.perf_counter_ns() - start
vid.release()
cv2.destroyAllWindows()

# posData = np.loadtxt('posData.csv', delimiter=',')
# timeData = np.loadtxt('timeData.csv', delimiter=',')


# np.savetxt('data/realPosData' + timestamp + '.csv', realPosData, delimiter=',')
# np.savetxt('data/posData' + timestamp + '.csv', posData, delimiter=',')
# np.savetxt('data/timeData' + timestamp + '.csv', timeData, delimiter=',')

posData = -(np.array(posData) - pos - 10)
realPosData = -(np.array(realPosData) - pos - 10)
mergedDopplers = [[-doppler for doppler in dopplers] for dopplers in mergedDopplers]
usedDopplers = [-doppler for doppler in usedDopplers]

import matplotlib.pyplot as plt
plt.plot(timeData, posData, 'b')
#error plt.plot(timeData, np.abs(np.array(posData) - np.array(realPosData)), 'r')
#plt.plot(timeData, realPosData, 'r')
plt.fill_between(timeData, realPosData - 0.5, realPosData + 0.5, facecolor='black')
splittedDopplers = [list(dopplers) for dopplers in zip(*mergedDopplers)]
colors = ('g', 'r', 'c', 'm', 'y')
for i, dopplers in enumerate(splittedDopplers):
    print(len(dopplers), len(timeData))
    plt.plot(timeData, dopplers, colors[i])
plt.plot(timeData, usedDopplers, 'orange')
#mostrar una curva por doppler (desviación)
plt.xlabel("Time (s)")
plt.ylabel("Position (cm)")
plt.title("Position over time")
#plt.yticks(np.arange(, 64, 2))
plt.grid()
fig = plt.gcf()
plt.show()

timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
description = input("Enter figure description:\n")
foldername = 'data/' + timestamp + "_" + description + "/"
import os
os.mkdir(foldername)
fig.savefig(foldername + "figure.png")
np.savetxt(foldername + "realPosData.csv", realPosData, delimiter=',')
np.savetxt(foldername + 'posData.csv', posData, delimiter=',')
np.savetxt(foldername + "timeData.csv", timeData, delimiter=',')
np.savetxt(foldername + "mergedDopplers.csv", mergedDopplers, delimiter=',')
np.savetxt(foldername + "usedDopplers.csv", usedDopplers, delimiter=',')

# dt = 0
# dL = 20 # Initial distance from left speaker to phone in cm
# vL = 0
# vR = 0
# dR = 20 # Initial distance from right speaker to phone in cm
# while True:
#     start = time.time()
#     data = sock.recv(2048) 
#     length = int.from_bytes(data[0:4], "big")

#     f.F = np.array([[1.,dt],
#                     [0.,1]])

#     import random
#     f.R = 0.0001
#     f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.000005)

#     if length == 1796:        
#         int_values = [x for x in data[4:length]]         
#         _, _, Sxx = signal.spectrogram(np.array(int_values), fs=44100, nfft=44100, nperseg=1792, mode='magnitude')
#         dopplerLS = getLSDoppler(Sxx)
#         #dopplerRS = getRSDoppler(Sxx)
#         dL = dL - dopplerLS*dt
#         #dR = dR - dopplerRS*dt
#         #print(dL)#TODO NOW debugguear si 1hz doppler = 2cm/s en f correspondiente
#         f.predict()
#         f.update(-dopplerLS*2)
#         print(f.x[0], f.x[1], dopplerLS, dt, dt*dopplerLS, dL)#TO
#         #print("DistanceLS: ", dL)
#         #print("DistanceRS: ", dR)
        
#         #print("DopplerLS: ", dopplerLS)
#         # ponderar best por Sxx (desindad spectral de potencia)
        
#         #print("indices of best", indicesOfBest)
        
#         #print("best dopplers: ", bestDopplers)
        
#         #print("best freqs: ", bestFreqs)
        
#         #dopplerRS = [np.argmax(Sxx[x-100:x+100]) - 100 for x in range(20000, 22000, 200)]
        
#         #print("DopplerRS: ", dopplerRS)

        
        
#         #if np.abs(np.mean(vLS)) > 3:
        
#         #vRS = (dopplerRS/21000)*346.6 * 100
        
#         #print("vLS: ", vLS)
#         #print("vRS: ", vRS)
#     else:
#         print("Invalid UDP packet.")
#     dt = time.time() - start