#from os import O_NONBLOCK
from pydoc import describe
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
    arr2 = np.c_[finalSamples, finalSamples] # Make stereo samples
    sound = pg.mixer.Sound(arr2)
    channel = 0 if speaker == 'L' else 1
    pgChannel = pg.mixer.Channel(channel)
    pgChannel.play(sound, -1)
    if channel == 0:
        pgChannel.set_volume(0.45, 0.0)
    else: 
        pgChannel.set_volume(0.0, 0.45)
    #pgChannel.set_volume(0.45, 0.45)

def remove_outliers(data, max_deviation=1.35):
    return (abs(data - np.median(data)) < max_deviation * np.std(data))

x = True
def getDoppler(Sxx, freqs, bremove_outliers=True):
    jump = freqs[2]
    hJ = int(jump/2.5)
    # Get displacement in Hz from original frequencies for each wave
    dopplerXS = np.array([np.argmax(Sxx[x-hJ:x+hJ]) - hJ for x in range(*freqs)])
    print(dopplerXS)
    not_outliers = np.ones(len(dopplerXS), dtype=bool)
    if bremove_outliers and not np.all(np.isclose(dopplerXS, dopplerXS[0])): # Do this check so it doesn't return an empty list
        not_outliers = remove_outliers(dopplerXS)
    
    # indicesOfBest = np.abs(dopplerXS).argsort()[:5][::-1] # Get the 
    # bestDopplers = dopplerXS[indicesOfBest]
    #ojo q se están borrando algunos...
    
    bestFreqs = np.arange((freqs[1] - freqs[0])/freqs[2]) * freqs[2] + freqs[0]
    
    vXS = np.array([(dopplerXS[i]/x) * 346.3 * 100 for i, x in enumerate(bestFreqs)])
    global x
    if x:
        mergedDopplers.append(vXS)
    x = not x
    return np.mean(vXS[not_outliers])

def getBlackSquareContour(contours, gray):
    maxArea = -1
    index = -1
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        mean = cv2.mean(gray[y:y+h, x:x+w])[0]
        area = cv2.contourArea(contour)

        if area > maxArea and mean < 80:
            index = i
            maxArea = area
    if index == -1:
        raise ValueError("Invalid contour index")
    return contours[index]

import cv2

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid.set(cv2.CAP_PROP_AUTOFOCUS, 0) 

lastPos = 1
posX = 77
posY = 10

realPosData = []
posDataL = []
posDataR = []
timeData = []

startTime = time.time()
cmPerPxX = -1
cmPerPxY = -1

objpoints = np.load('objpoints.csv.npy')
imgpoints = np.load('imgpoints.csv.npy')


print(objpoints)
print(imgpoints)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (640, 480), None, None)

ret, img = vid.read()
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))


#Left speaker tones

lFreqs = (18000, 19000, 200)
rFreqs = (19000, 20000, 200)

playSound(*lFreqs, 'L')
playSound(*rFreqs, 'R')

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
dL = posX # Initial distance from left speaker to phone in cm
vL = 0
vR = 0
dR = 77 # Initial distance from right speaker to phone in cm
init = time.time()
timestamp = time.strftime("%d-%m-%Y-%H:%M:%S")

while True:
    start = time.time()
    data = sock.recv(2048)     
    length = int.from_bytes(data[0:4], "big")
    
    if length == 1796:        
        if time.time() - 1 < init:
            continue
        int_values = [x for x in data[4:length]]         
        _, _, Sxx = signal.spectrogram(np.array(int_values), fs=44100, nfft=44100, nperseg=1792, mode='magnitude')
        
        dopplerLS = getDoppler(Sxx, lFreqs)
        dopplerRS = getDoppler(Sxx, rFreqs)
        dL += dopplerLS * dt
        dR += dopplerRS * dt
        
        usedDopplers.append(dopplerLS)
        #dR = dR - dopplerRS*dt
        #TODO NOW debugguear si 1hz doppler = 2cm/s en f correspondiente

        start2 = time.time()
        ret, frame = vid.read()
        
        # dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        
        ret, binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
        eroded = binary.copy()
        cv2.erode(binary, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1]), eroded)
        
        contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blackSquareContour = getBlackSquareContour(contours, gray)
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)    
        x, y, w, h = cv2.boundingRect(blackSquareContour)
        x = int(x + w/2)
        if cmPerPxX == -1:
            cmPerPxX = 17.3/w
            camPosX = x
            camPosY = y
        
        cmPerPxY = 0.125
        print(cmPerPxY)         
        realPosX = (x-camPosX)*cmPerPxX + posX
        realPosY = -(y-camPosY)*cmPerPxY + posY
        print(dL, dR, realPosX, realPosY)
        

        #print(x, y, -(x-582)*cmPerPx + pos)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("img", frame)
        cv2.waitKey(1)
        dt = time.time() - start2
        posDataL.append(dL)
        posDataR.append(dR)
        realPosData.append(realPosX)
        timeData.append(time.time() - startTime)
        
    
        #speed = ((x - lastPos)*cmPerPx)/dt
        #print("Speed = ", speed, " cm/s")
        lastPos = x
        import keyboard
        if keyboard.is_pressed('q'):
            break
    #print("FPS: ", 1/(time.time() - start))
    dt = time.time() - start
vid.release()
cv2.destroyAllWindows()

pg.mixer.stop()
# posData = np.loadtxt('posData.csv', delimiter=',')
# timeData = np.loadtxt('timeData.csv', delimiter=',')


# np.savetxt('data/realPosData' + timestamp + '.csv', realPosData, delimiter=',')
# np.savetxt('data/posData' + timestamp + '.csv', posData, delimiter=',')
# np.savetxt('data/timeData' + timestamp + '.csv', timeData, delimiter=',')

posDataL = -(np.array(posDataL) - pos - 10)
posDataR = -(np.array(posDataR) - pos - 10)
realPosData = -(np.array(realPosData) - pos - 10)
mergedDopplers = [[-doppler for doppler in dopplers] for dopplers in mergedDopplers]
usedDopplers = [-doppler for doppler in usedDopplers]

import matplotlib.pyplot as plt
plt.plot(timeData, posDataL, 'b')
plt.plot(timeData, posDataR, 'lime')
#error plt.plot(timeData, np.abs(np.array(posData) - np.array(realPosData)), 'r')
#plt.plot(timeData, realPosData, 'r')
plt.fill_between(timeData, realPosData - 0.5, realPosData + 0.5, facecolor='black')
splittedDopplers = [list(dopplers) for dopplers in zip(*mergedDopplers)]
colors = ('g', 'r', 'c', 'm', 'y')
for i, dopplers in enumerate(splittedDopplers):
    plt.plot(timeData, dopplers)
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
np.savetxt(foldername + 'posDataL.csv', posDataL, delimiter=',')
np.savetxt(foldername + 'posDataR.csv', posDataR, delimiter=',')
np.savetxt(foldername + "timeData.csv", timeData, delimiter=',')
np.savetxt(foldername + "mergedDopplers.csv", mergedDopplers, delimiter=',')
np.savetxt(foldername + "usedDopplers.csv", usedDopplers, delimiter=',')