import numpy as np
from scipy import signal, special
from plotter import *

class DopplerAnalyzer:
    ID = -1

    def __init__(self, frequencies) -> None:
        self.id = DopplerAnalyzer.ID
        DopplerAnalyzer.ID += 1
        self.all_frequency_displacements = [0 for f in frequencies]
        self.frequencies = frequencies

    def extract_speeds_from(self, audio_samples):
        plotter.add_sample('audio_samples', audio_samples)
        _, _, Sxx = signal.spectrogram(audio_samples, fs=44100, nfft=44100, nperseg=1792, mode='magnitude')
        
        speed = self.extract_speed_from(Sxx, np.array(self.frequencies))
        
        plotter.add_sample(f'doppler_deviation_filtered_{self.id}', -speed)
        
        return speed

    def extract_speed_from(self, Sxx, frequencies):
        flw = 100 # Frequency lookup width
        
        # Get displacement in Hz from original frequencies for each wave
        frequency_displacements = np.array([np.argmax(Sxx[f-flw:f+flw]) - flw for f in frequencies])
#        np.sum(np.square(frequency_displacements - mean_freqs_displacements))
        self.all_frequency_displacements.append(frequency_displacements)
        difference = np.abs(self.all_frequency_displacements[-1] - frequency_displacements)
        greater_than_ten = difference > 10
        if np.all(greater_than_ten):
            frequency_displacements.fill(self.all_frequency_displacements[np.argmin(difference)])
        else: 
            frequency_displacements[greater_than_ten] = np.mean(frequency_displacements)
        variances = np.var(self.all_frequency_displacements, axis=0, ddof=1)
        
        variances[variances == 0] = 0.00001
        
        # Plot
        # speeds = np.array([(frequency_displacements[i]/frequency) * 346.3 * 100 for i, frequency in enumerate(frequencies)]) 
        # for i, frequency in enumerate(frequencies):
        #     plotter.add_sample(f'doppler_deviation_{frequency}_hz', frequency_displacements[i])
        ###

        frequency_displacements, frequencies, variances = DopplerAnalyzer.filter_frequencies(frequency_displacements, frequencies, variances)
        
        # Apply Doppler effect formula to compute speed in cm/s
        speeds = np.array([(frequency_displacements[i]/frequency) * 346.3 * 100 for i, frequency in enumerate(frequencies)])

        #variances = 1/variances
        mean = np.sum(speeds * (variances/np.sum(variances)))
        #TODO take into account that higher frequencies mean more speed
        
        
        return mean

    @staticmethod
    def filter_frequencies(frequency_displacements, frequencies, variances, remove_outliers=True):
        if remove_outliers:
            not_outliers = ~DopplerAnalyzer.find_outliers(frequency_displacements)
            frequency_displacements = frequency_displacements[not_outliers]
            frequencies = frequencies[not_outliers]
            variances = variances[not_outliers]
        frequency_displacements[np.abs(frequency_displacements) <= 1.5] = 0 # Treshold to avoid small noise
        best_frequencies_indices = DopplerAnalyzer.select_best_frequencies(frequency_displacements)
        frequency_displacements = frequency_displacements[best_frequencies_indices]
        frequencies = frequencies[best_frequencies_indices]
        variances = variances[best_frequencies_indices]
        return frequency_displacements, frequencies, variances
    

    @staticmethod
    def select_best_frequencies(frequency_displacements):
        # indicesOfBest = np.abs(dopplerXS).argsort()[:5][::-1] # Get the 
        return np.ones(len(frequency_displacements), dtype=bool) # TODO replace by actual algorithm
    
    @staticmethod
    def find_outliers(data, max_deviation=2.35): 
        if not np.all(np.isclose(data, data[0])):  # Do this check so it doesn't return an empty list
            return np.array(abs(data - np.median(data)) > max_deviation * np.std(data), dtype=bool)
        else:
            return np.zeros(len(data), dtype=bool)