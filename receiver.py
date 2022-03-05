import socket
import time
import numpy as np

class Receiver:
    def __init__(self, port=5555):
        self.socket = socket.socket(socket.AF_INET, # Internet
                                    socket.SOCK_DGRAM) # UDP
        ip_address = Receiver.get_pc_ip()
        self.socket.bind((ip_address, port))
        print("Listening on: ", ip_address, ":", port)

        # Discard first packets because they are noisy
        end_time = time.time() + 0.5
        while time.time() < end_time:
            self.socket.recv(2048)

    def retrieve_sound_samples(self):
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