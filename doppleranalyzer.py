import numpy as np
from scipy import signal
from plotter import *

class DopplerAnalyzer:

    @staticmethod
    def extract_speeds_from(audio_samples, all_frequencies):
        _, _, Sxx = signal.spectrogram(audio_samples, fs=44100, nfft=44100, nperseg=1792, mode='magnitude')
        speeds = []
        for frequencies in all_frequencies:
            speed = DopplerAnalyzer.extract_speed_from(Sxx, frequencies)
            speeds.append(speed)
        return np.array(speeds)

    @staticmethod
    def extract_speed_from(Sxx, frequencies):
        flw = 100 # Frequency lookup width
        
        # Get displacement in Hz from original frequencies for each wave
        frequency_displacements = np.array([np.argmax(Sxx[f-flw:f+flw]) - flw for f in frequencies])
        
        frequency_displacements = DopplerAnalyzer.filter_frequencies(frequency_displacements)

        # Apply Doppler effect formula to compute speed in cm/s
        speeds = np.array([(frequency_displacements[i]/frequency) * 346.3 * 100 for i, frequency in enumerate(frequencies)]) 
        for i, frequency in enumerate(frequencies):
            plotter.add_sample(f'doppler_deviation_{frequency}_hz', speeds[i])

        mean = np.mean(speeds)
        plotter.add_sample(f'filtered_deviation_chosen', mean)
        return mean

    @staticmethod
    def filter_frequencies(frequency_displacements, remove_outliers=True):
        frequency_displacements[np.abs(frequency_displacements) <= 1] = 0 # Treshold to avoid small noise
        
        if remove_outliers:
            not_outliers = not DopplerAnalyzer.find_outliers(frequency_displacements)
            frequency_displacements = frequency_displacements[not_outliers]
    
        best_frequencies = DopplerAnalyzer.select_best_frequencies(frequency_displacements)
        return best_frequencies
    

    @staticmethod
    def select_best_frequencies(frequency_displacements):
        # indicesOfBest = np.abs(dopplerXS).argsort()[:5][::-1] # Get the 
        return frequency_displacements # TODO replace by actual algorithm
    
    @staticmethod
    def find_outliers(data, max_deviation=1.35): 
        if not np.all(np.isclose(data, data[0])):  # Do this check so it doesn't return an empty list
            return (abs(data - np.median(data)) > max_deviation * np.std(data))
        else:
            return np.zeros(len(data), dtype=bool)