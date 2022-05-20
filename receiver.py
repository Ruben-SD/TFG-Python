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
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
        ip_address = Receiver.get_pc_ip()
        self.socket.bind((ip_address, port))
        print("Listening on: ", ip_address, ":", port)

        # Discard first packets because they are noisy
        end_time = time.time() + 3
        while time.time() < end_time:
            self.socket.recv(2048)

        pa = pyaudio.PyAudio()
        FORMAT = pyaudio.paInt16

        CHANNELS = 1
        RATE = 44100

        self.stream = pa.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              frames_per_buffer=1792)

    def read_phone_mic(self):
        data = self.socket.recv(2048)
        length = int.from_bytes(data[0:4], "big")
        if length != 1796:
            raise ValueError("Received malformed packet")
        int_values = np.array([x for x in data[4:length]])
        return int_values

    def read_pc_mic(self):
        frames = [] # A python-list of chunks(numpy.ndarray)
        for _ in range(0, int(44100 / 1792 * 10)):
            print("RECORDING")
            data = self.stream.read(1792)
            frames.append(np.fromstring(data, dtype=np.int16))

        #Convert the list of numpy-arrays into a 1D array (column-wise)
        numpydata = np.hstack(frames)
        from scipy.io.wavfile import write
        write('test.wav', 44100, numpydata)
        print("DONE")
        return "asd"

    @staticmethod
    def get_pc_ip():
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_socket.connect(("8.8.8.8", 80))
        pc_ip_address = temp_socket.getsockname()[0]
        temp_socket.close()
        return pc_ip_address
