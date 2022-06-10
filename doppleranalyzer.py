import numpy as np
from scipy import signal

class DopplerAnalyzer:
    ID = 0

    def __init__(self, frequencies, plotter, config) -> None:
        self.options = config['options'] if "options" in config else None
        self.plotter = plotter
        self.id = DopplerAnalyzer.ID
        DopplerAnalyzer.ID += 1
        self.all_frequency_displacements = [list(np.zeros(len(frequencies)))]
        self.frequencies = frequencies

    def extract_speeds_from(self, audio_samples, cosine):
        _, _, Sxx = signal.spectrogram(audio_samples, fs=44100, nfft=44100, nperseg=len(audio_samples), mode='magnitude')
        
        speed = self.extract_speed_from(Sxx, np.array(self.frequencies), cosine)
        
        self.plotter.add_sample(f'Doppler_deviation_filtered_{self.id}', speed)
        
        return speed

    def extract_speed_from(self, Sxx, frequencies, cosine):
        flw = 100 # Frequency lookup width
        
        # Get displacement in Hz from original frequencies for each wave
        frequency_displacements = np.array([np.argmax(Sxx[f-flw:f+flw]) - flw for f in frequencies])
#        np.sum(np.square(frequency_displacements - mean_freqs_displacements))

        self.all_frequency_displacements.append(frequency_displacements)
  

        # Plot
        ###

        #frequency_displacements, frequencies, variances = self.filter_frequencies(frequency_displacements, frequencies, variances=variances, remove_outliers=not self.options or 'outlier_removal' in self.options)
        
        # Apply Doppler effect formula to compute speed in cm/s
        cosine = max(cosine, 0.2)
        speeds = np.array([(frequency_displacements[i]/(frequency*cosine)) * 343.73 * 100 for i, frequency in enumerate(frequencies)])
        print(cosine, speeds)

        #variances = 1/variances
        mean = np.mean(speeds)
        # weights = [np.argmax(Sxx[f-flw:f+flw]) for f in frequencies]
        # mean = np.average(frequency_displacements, weights=weights)
        #TODO take into account that higher frequencies mean more speed
        
        
        return mean

    def filter_frequencies(self, frequency_displacements, frequencies, variances=None, remove_outliers=True):
        if remove_outliers:
            if self.options and 'outlier_removal' in self.options:
                max_deviation = self.options['outlier_removal']['values'][self.options['outlier_removal']['index']]
            else: 
                max_deviation = 1.5
            not_outliers = ~DopplerAnalyzer.find_outliers(frequency_displacements, max_deviation=max_deviation)
            frequency_displacements = frequency_displacements[not_outliers]
            frequencies = frequencies[not_outliers]
            if variances is not None:
                variances = variances[not_outliers]
        
        if not self.options:
            frequency_displacements[np.abs(frequency_displacements) <= 1.5] = 0 # Treshold to avoid small noise
        elif 'doppler_threshold' in self.options:
            threshold = self.options['doppler_threshold']['values'][self.options['doppler_threshold']['index']]
            frequency_displacements[np.abs(frequency_displacements) <= threshold] = 0
            
        best_frequencies_indices = DopplerAnalyzer.select_best_frequencies(frequency_displacements)
        frequency_displacements = frequency_displacements[best_frequencies_indices]
        frequencies = frequencies[best_frequencies_indices]
        if variances is not None:
            variances = variances[best_frequencies_indices]
        return frequency_displacements, frequencies, variances
    

    @staticmethod
    def select_best_frequencies(frequency_displacements):
        #indicesOfBest = np.abs(frequency_displacements).argsort()[:2][::-1] # Get the 
        return np.ones(len(frequency_displacements), dtype=bool) # TODO replace by actual algorithm
    
    @staticmethod
    def find_outliers(data, max_deviation=1.35): 
        return np.array(abs(data - np.median(data)) > max_deviation * np.std(data), dtype=bool)
