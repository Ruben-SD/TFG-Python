import numpy as np
from scipy import signal
from plotter import *

class DopplerAnalyzer:

    @staticmethod
    def extract_speeds_from(audio_samples, all_frequencies):
        _, _, Sxx = signal.spectrogram(audio_samples, fs=44100, nfft=44100, nperseg=1792, mode='magnitude')
        speeds = []
        for i, frequencies in enumerate(all_frequencies):
            speed = DopplerAnalyzer.extract_speed_from(Sxx, np.array(frequencies))
            plotter.add_sample(f'doppler_deviation_filtered_{i}', -speed)
            speeds.append(speed)
        return np.array(speeds)

    @staticmethod
    def extract_speed_from(Sxx, frequencies):
        flw = 100 # Frequency lookup width
        
        # Get displacement in Hz from original frequencies for each wave
        frequency_displacements = np.array([np.argmax(Sxx[f-flw:f+flw]) - flw for f in frequencies])
        
        # Plot
        # speeds = np.array([(frequency_displacements[i]/frequency) * 346.3 * 100 for i, frequency in enumerate(frequencies)]) 
        # for i, frequency in enumerate(frequencies):
        #     plotter.add_sample(f'doppler_deviation_{frequency}_hz', -speeds[i])
        ###

        frequency_displacements, frequencies = DopplerAnalyzer.filter_frequencies(frequency_displacements, frequencies)

        # Apply Doppler effect formula to compute speed in cm/s
        speeds = np.array([(frequency_displacements[i]/frequency) * 346.3 * 100 for i, frequency in enumerate(frequencies)]) 
        
        mean = np.mean(speeds)
        
        return mean

    @staticmethod
    def filter_frequencies(frequency_displacements, frequencies, remove_outliers=True):
        if remove_outliers:
            not_outliers = ~DopplerAnalyzer.find_outliers(frequency_displacements)
            frequency_displacements = frequency_displacements[not_outliers]
            frequencies = frequencies[not_outliers]
        frequency_displacements[np.abs(frequency_displacements) <= 1] = 0 # Treshold to avoid small noise
        best_frequencies_indices = DopplerAnalyzer.select_best_frequencies(frequency_displacements)
        frequency_displacements = frequency_displacements[best_frequencies_indices]
        frequencies = frequencies[best_frequencies_indices]
        return frequency_displacements, frequencies
    

    @staticmethod
    def select_best_frequencies(frequency_displacements):
        # indicesOfBest = np.abs(dopplerXS).argsort()[:5][::-1] # Get the 
        return np.ones(len(frequency_displacements), dtype=bool) # TODO replace by actual algorithm
    
    @staticmethod
    def find_outliers(data, max_deviation=1.35): 
        if not np.all(np.isclose(data, data[0])):  # Do this check so it doesn't return an empty list
            return np.array(abs(data - np.median(data)) > max_deviation * np.std(data), dtype=bool)
        else:
            return np.zeros(len(data), dtype=bool)