import socket
import time
import numpy as np
import pyaudio
import struct
SHORT_NORMALIZE = (1.0/32768.0)


class Receiver:
    def __init__(self, port=5555):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ip_address = Receiver.get_pc_ip()
        print("Listening on: ", ip_address, ":", port)
        self.socket.bind((ip_address, port))
        self.socket.listen()
        print("Waiting for connection...")
        self.conn, addr = self.socket.accept()
        print(f"Connected to {addr}")        

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
        end_time = time.time() + 3
        while time.time() < end_time:
            self.conn.recv(1796, socket.MSG_WAITALL)
            # length = int.from_bytes(data[0:4], "big")
            # print(length)
        #end_time = time.time() + 3
        # while time.time() < end_time:
        #     self.stream.read(1792)
        print("Finished discarding")
        self.last = None


    def read_phone_mic(self):
        print("R")
        data = self.conn.recv(1796, socket.MSG_WAITALL)
        print("C")
        length = int.from_bytes(data[0:4], "big")
        
        if self.last is not None:       
            if length != self.last + 1:
                print(length, self.last + 1)     
                raise ValueError("Received out of order packet")
        self.last = length

        if len(data) != 1792 + 4: # * 2 + 4
            raise ValueError(f"Received malformed packet of length {len(data)}")
        
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
