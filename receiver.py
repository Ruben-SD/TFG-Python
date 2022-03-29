import socket
import time
import numpy as np
import array

class Receiver:
    def __init__(self, port=5555):
        self.socket = socket.socket(socket.AF_INET, # Internet
                                    socket.SOCK_DGRAM) # UDP
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
        ip_address = Receiver.get_pc_ip()
        self.socket.bind((ip_address, port))
        print("Listening on: ", ip_address, ":", port)

        # Discard first packets because they are noisy
        end_time = time.time() + 1
        
        while time.time() < end_time:
            self.socket.recv(1921 * 2)

    def retrieve_sound_samples(self):
        # import pygame
        # pygame.mixer.stop()
        # total_samples = []
        # while True:
        data = self.socket.recv(1921 * 2)
        data = array.array('h', data)
        data.byteswap()
                
        if len(data) != 1921 or data[0] != 342:
            print(len(data))
            raise ValueError("Received malformed packet")

        samples = np.array(data[1:].tolist())
        # total_samples.append(samples)
        # print(len(total_samples))
        # if len(total_samples) == 48000/1920 * 5:
        #     break
    
        import sounddevice as sd
       
        x = np.array(samples).flatten()
        # print(len(x))
        # x = np.interp(x, (x.min(), x.max()), (-1, +1))
        # print(x)
        # sd.play(x[1::2], 48000, blocking=True)
        
        l = x[::2]
        r = x[1::2]

        result = (l + r) / 2

        return np.array(result)

    @staticmethod
    def get_pc_ip():
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_socket.connect(("8.8.8.8", 80))
        pc_ip_address = temp_socket.getsockname()[0]
        temp_socket.close()
        return pc_ip_address