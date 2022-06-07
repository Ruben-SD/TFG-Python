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
            self.socket.recv(2048)
        end_time = time.time() + 3
        # while time.time() < end_time:
        #     self.stream.read(1792)
        print("Finished discarding")


    def read_phone_mic(self):
        #length = int.from_bytes(data[0:4], "big")

        import sounddevice as sd

        fs = 44100
        i = 0
        while i < 50:
            i = i + 1
            self.socket.recv(2048)
        
        x = []
        # if length != 1796:
        #     raise ValueError("Received malformed packet")
        last = None
        while len(x) < 44100 * 5:
            
            data = self.socket.recv(2048)
            length = int.from_bytes(data[0:4], "big")
            if last is not None and length != last + 1:
                raise Exception(last, length)
            last = length
            print("Reading", length)
            int_values = [x for x in data[4:1796]]
            x = x + int_values
        x = np.array(x)
        
        x = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
        print(x, len(x))
        print("Playing")
        sd.play(x, 44100)
        sd.wait()
        print("dONE")
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
