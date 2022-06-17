import socket
import time
import numpy as np
import pyaudio
import struct
SHORT_NORMALIZE = (1.0/32768.0)


class Receiver:
    def __init__(self, port=5555):
        self.socket = socket.socket(socket.AF_INET,  # Internet
                                    socket.SOCK_DGRAM)  # UDP
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 3588*24)
        ip_address = Receiver.get_pc_ip()
        self.socket.bind((ip_address, port))
        print("Listening on: ", ip_address, ":", port)

        pa = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16

        CHANNELS = 1
        RATE = 44100

        self.stream = pa.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=1792)

        # Discard first packets because they are noisy
        print("Discarding initial noisy audio samples...")
        end_time = time.time() + 5
        while time.time() < end_time:
            self.socket.recv(3588)
        end_time = time.time() + 3
        # while time.time() < end_time:
        #     self.stream.read(1792)
        print("Finished discarding")


    def read_phone_mic(self):
        data = self.socket.recv(3588)
        length = int.from_bytes(data[0:4], "big")
        
        if len(data) != 1792  + 4: # * 2 + 4
            raise ValueError("Received malformed packet")
        
        int_values = np.frombuffer(data[4:], dtype=np.int8)#np.array([x for x in data[4:len(data)]])
        # print(len(int_values))
        # print(int_values)
        return int_values

    def read_pc_mic(self):
        data = self.stream.read(1792)
        ret = np.frombuffer(data, dtype=np.int16)
        # import sounddevice as sd

        # fs = 44100
        # sd.play(ret, fs)

        return ret

    @staticmethod
    def get_pc_ip():
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_socket.connect(("8.8.8.8", 80))
        pc_ip_address = temp_socket.getsockname()[0]
        temp_socket.close()
        return pc_ip_address
