import numpy as np
from scipy import signal
from plotter import *

class DopplerAnalyzer:

    @staticmethod
    def get_speeds_from(audio_samples, all_frequencies):
        _, _, Sxx = signal.spectrogram(audio_samples, fs=44100, nfft=44100, nperseg=1792, mode='magnitude')
        speeds = []
        for frequencies in all_frequencies:
            speed = DopplerAnalyzer.get_speed_of(Sxx, frequencies)
            speeds.append(speed)
        return np.array(speeds)

    @staticmethod
    def get_speed_of(Sxx, frequencies, bremove_outliers=True):
        flw = 100 # Frequency lookup width
        
        # Get displacement in Hz from original frequencies for each wave
        frequency_displacements = np.array([np.argmax(Sxx[x-flw:x+flw]) - flw for x in frequencies])
        frequency_displacements[np.abs(frequency_displacements) <= 1] = 0
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
        for i, frequency in enumerate(frequencies):
            plotter.add_sample(f'doppler_deviation_{frequency}_hz', speeds[i])

        #get_bests() clases para cada tipo de selección
        mean = np.mean(speeds[not_outliers])
        plotter.add_sample(f'doppler_deviation_chosen', mean)
        return mean

    @staticmethod
    def remove_outliers(data, max_deviation=1.35):
        return (abs(data - np.median(data)) < max_deviation * np.std(data))