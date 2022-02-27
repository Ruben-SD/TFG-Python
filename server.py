import socket
from scipy import signal
import numpy as np
import pygame
import time
import cv2
import json

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
        channel.set_volume(0.45, 0)

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
        final_samples = np.c_[samples, samples] # Make stereo samples (Sound() expected format)
        return final_samples

    def get_config(self):
        return self.config

class Receiver:
    def __init__(self, port=5555):
        self.socket = socket.socket(socket.AF_INET, # Internet
                                    socket.SOCK_DGRAM) # UDP
        ip_address = Receiver.get_pc_ip()
        self.socket.bind((ip_address, port))
        print("Listening on: ", ip_address, ":", port)

        # Discard first packets because they are noisy
        end_time = time.time() + 1.5
        while time.time() < end_time:
            self.socket.recv(2048)

    def read_packet(self):
        data = self.socket.recv(2048)
        length = int.from_bytes(data[0:4], "big")
        if length != 1796:
            raise ValueError("Received malformed packet")
        int_values = np.array([x for x in data[4:length]])
        return int_values

    @staticmethod
    def get_pc_ip():
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_socket.connect(("8.8.8.8", 80))
        pc_ip_address = temp_socket.getsockname()[0]
        temp_socket.close()
        return pc_ip_address

class Positioner:
    def __init__(self):
        position_config = config['smartphone']['position'] 
        self.initial_position = position_config['x'], position_config['y']
        self.position = self.initial_position

    def set_position(self, position):
        self.position = tuple([sum(x) for x in zip(self.initial_position, position)])

    def move_by(self, amount):
        self.position = self.position[0] + amount[0], self.position[1] + amount[1]

    def update_position(self, dt):
        pass


class Predictor(Positioner):
    def __init__(self, config):
        super().__init__()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.speakers = [Speaker(speaker_config) for speaker_config in config['speakers']]
        for speaker in self.speakers:
            speaker.play_sound()
        self.receiver = Receiver()

    #TODO abstract to update_measurement
    def update_position(self, dt):
        sound_samples = self.receiver.read_packet()
        speeds = DopplerAnalyzer.get_speeds_from(sound_samples, [speaker.get_config().get_frequencies() for speaker in self.speakers])#, 
        # asbtract to measurement
        #self.position.add_speed(vx * dt, vy * dt)
        print(speeds)
#todo   
        self.position = (self.position[0] + speeds[0] * dt, 0)
        return self.position

    def __del__(self):
        pygame.mixer.stop()

        
class DopplerAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def get_speeds_from(audio_samples, all_frequencies):
        _, _, Sxx = signal.spectrogram(audio_samples, fs=44100, nfft=44100, nperseg=1792, mode='magnitude')
        speeds = []
        for frequencies in all_frequencies:
            dopplers = DopplerAnalyzer.get_doppler_of(Sxx, frequencies) # returns speed
            #speed = DopplerAnalyzer.get_speed_from(dopplers)
            speeds.append(dopplers)
        return speeds

    @staticmethod
    def get_doppler_of(Sxx, frequencies, bremove_outliers=True):
        flw = 100 # Frequency lookup width
        
        # Get displacement in Hz from original frequencies for each wave
        frequency_displacements = np.array([np.argmax(Sxx[x-flw:x+flw]) - flw for x in frequencies])
        
        not_outliers = np.ones(len(frequency_displacements), dtype=bool)
        if bremove_outliers and not np.all(np.isclose(frequency_displacements, frequency_displacements[0])): # Do this check so it doesn't return an empty list
            not_outliers = DopplerAnalyzer.remove_outliers(frequency_displacements)
        
        # indicesOfBest = np.abs(dopplerXS).argsort()[:5][::-1] # Get the 
        # bestDopplers = dopplerXS[indicesOfBest]
        #ojo q se están borrando algunos...
        #filter 1Hz?
        
        #TODO
        #bestFreqs = np.arange((freqs[1] - freqs[0])/freqs[2]) * freqs[2] + freqs[0]
        
        # Doppler effect formula to compute speed in cm/s
        speeds = np.array([(frequency_displacements[i]/frequency) * 346.3 * 100 for i, frequency in enumerate(frequencies)]) # select best frequencies
        
        #get_bests() clases para cada tipo de selección
        return np.mean(speeds[not_outliers])

    #def get_speed_from(self, dopplers):



    @staticmethod
    def remove_outliers(data, max_deviation=1.35):
        return (abs(data - np.median(data)) < max_deviation * np.std(data))


class CameraSystem(Positioner):
    def __init__(self, config):
        super().__init__()
        self.SMARTPHONE_WIDTH_CM = config['smartphone']['dims']['length']
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
        _, first_frame = self.cam.read()
        self.cm_per_width_px = self.get_cm_per_smartphone_px_width(first_frame)
        self.initial_smartphone_cam_pos, _ = self.get_smartphone_img_coords(first_frame)

    def update_position(self, dt):
        x = self.get_smartphone_world_position()
        self.set_position((x, 0))
        return self.position

    def get_smartphone_world_position(self):
        _, frame = self.cam.read()
        (x, y) = self.get_smartphone_img_coords(frame)  
        current_position = (x - self.initial_smartphone_cam_pos) * self.cm_per_width_px
        return current_position #x, ycalc))

    def get_smartphone_img_coords(self, frame):
        x, y, w, h = self.get_smartphone_bounding_rect(frame)
        x = int(x + w/2)
        y = int(y + h/2)
        return (x, y)

    def get_smartphone_dims(self, frame):
        _, _, w, h = self.get_smartphone_bounding_rect(frame)
        return (w, h)

    def get_smartphone_bounding_rect(self, frame):
        binary = self.binarize_image(frame)
        improved = self.improve_binary_img(binary)
        contours, _ = cv2.findContours(improved, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        smartphone_contour = self.get_smartphone_contour(contours, frame)
        x, y, w, h = cv2.boundingRect(smartphone_contour)
        return (x, y, w, h)

    def binarize_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
        _, binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
        return binary

    def improve_binary_img(self, binary):
        eroded = cv2.erode(binary, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1]))
        return eroded

    @staticmethod #TODO Could take into consideration smartphone dimensions to improve detection
    def get_smartphone_contour(contours, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
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
            raise ValueError("Cannot find smartphone shaped black contour in image")
        return contours[index]

    def get_cm_per_smartphone_px_width(self, img):
        width, _ = self.get_smartphone_dims(img)
        return self.SMARTPHONE_WIDTH_CM/width

    def __del__(self):
        self.cam.release()
    
# class Evaluator:
#     def __init__(self) -> None:
#         self.ground_truth_system = Camera2D()
#         self.

#     def get_measurement:

#     def 

class FrameTimer:
    def __init__(self):
        self.last_frame_time = time.time()

    def mark(self):
        delta_time = time.time() - self.last_frame_time
        #print("FPS: ", 1/delta_time)
        self.last_frame_time = time.time()
        return delta_time


with open('config.json', 'r') as f:
    config = json.load(f)

frame_timer = FrameTimer()

predictor = Predictor(config)
ground_truth = CameraSystem(config)

while True:
    delta_time = frame_timer.mark()

    predicted_position = predictor.update_position(delta_time)
    real_position = ground_truth.update_position(delta_time)
    
    print(f" Predicted position: {predicted_position} Real position: {real_position}")
    
    # command = gui.update(tracker.get_visualization())
    
    # if command == plot:
    #     plotter.show(timeData, predictor.get_position(), tracker.get_position())
    # elif command == save:
    #     data_serializer.save(timeData, predictor.get_all_positions(), tracker.get_all_positions(), gui.ask_description())
    # elif command == exit:
    #     sys.exit(1)

    #

del receptor
del predictor
del ground_truth
# posData = np.loadtxt('posData.csv', delimiter=',')
# timeData = np.loadtxt('timeData.csv', delimiter=',')


# np.savetxt('data/realPosData' + timestamp + '.csv', realPosData, delimiter=',')
# np.savetxt('data/posData' + timestamp + '.csv', posData, delimiter=',')
# np.savetxt('data/timeData' + timestamp + '.csv', timeData, delimiter=',')

# posDataL = -(np.array(posDataL) - pos - 10)
# posDataR = -(np.array(posDataR) - pos - 10)
# realPosData = -(np.array(realPosData) - pos - 10)
# mergedDopplers = [[-doppler for doppler in dopplers] for dopplers in mergedDopplers]
# usedDopplers = [-doppler for doppler in usedDopplers]

# import matplotlib.pyplot as plt
# plt.plot(timeData, posDataL, 'b')
# plt.plot(timeData, posDataR, 'lime')
# #error plt.plot(timeData, np.abs(np.array(posData) - np.array(realPosData)), 'r')
# #plt.plot(timeData, realPosData, 'r')
# plt.fill_between(timeData, realPosData - 0.5, realPosData + 0.5, facecolor='black')
# splittedDopplers = [list(dopplers) for dopplers in zip(*mergedDopplers)]
# colors = ('g', 'r', 'c', 'm', 'y')
# for i, dopplers in enumerate(splittedDopplers):
#     plt.plot(timeData, dopplers)
# plt.plot(timeData, usedDopplers, 'orange')
# #mostrar una curva por doppler (desviación)
# plt.xlabel("Time (s)")
# plt.ylabel("Position (cm)")
# plt.title("Position over time")
# #plt.yticks(np.arange(, 64, 2))
# plt.grid()
# fig = plt.gcf()
# plt.show()

# timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
# description = input("Enter figure description:\n")
# foldername = 'data/' + timestamp + "_" + description + "/"
# import os
# os.mkdir(foldername)
# fig.savefig(foldername + "figure.png")
# np.savetxt(foldername + "realPosData.csv", realPosData, delimiter=',')
# np.savetxt(foldername + 'posDataL.csv', posDataL, delimiter=',')
# np.savetxt(foldername + 'posDataR.csv', posDataR, delimiter=',')
# np.savetxt(foldername + "timeData.csv", timeData, delimiter=',')
# np.savetxt(foldername + "mergedDopplers.csv", mergedDopplers, delimiter=',')
# np.savetxt(foldername + "usedDopplers.csv", usedDopplers, delimiter=',')