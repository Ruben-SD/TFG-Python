import numpy as np
from scipy import signal
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class DopplerAnalyzer:
    ID = 0

    def __init__(self, frequencies, plotter, config) -> None:
        self.options = config['options'] if "options" in config else None
        self.plotter = plotter
        self.id = DopplerAnalyzer.ID
        DopplerAnalyzer.ID += 1
        self.all_frequency_displacements = [list(np.zeros(len(frequencies)))]
        self.frequencies = frequencies

        self.my_filter = KalmanFilter(dim_x=2, dim_z=1)
        self.my_filter.x = np.array([[0.],
                [0.]])       # initial state (location and velocity)

        self.my_filter.F = np.array([[1.,1.],
                        [0.,1.]])    # state transition matrix

        self.my_filter.H = np.array([[0.,1.]])    # Measurement function
        self.my_filter.P *= 10                 # covariance matrix
        self.my_filter.R = 0.001                      # state uncertainty
        self.my_filter.Q = Q_discrete_white_noise(2, 1/24.0, .1) # process uncertainty

    def extract_speeds_from(self, audio_samples, cosine):
        _, _, Sxx = signal.spectrogram(audio_samples, fs=44100, nfft=44100, nperseg=len(audio_samples), mode='magnitude')
        
        speed = self.extract_speed_from(Sxx, np.array(self.frequencies), cosine)
        
        self.plotter.add_sample(f'Doppler_deviation_filtered_{self.id}', speed)

        self.my_filter.predict()
        self.my_filter.update(speed)

        # self.plotter.add_sample(f'Doppler_deviation_filtered_{self.id}_K', self.my_filter.x[1][0])
        if 'kalman_filter' in self.options:
            return self.my_filter.x[1][0]
        
        return speed

    def extract_speed_from(self, Sxx, frequencies, cosine):
        # Frequency lookup width
        flw = self.options['frequency_lookup_width'] if self.options is not None and 'frequency_lookup_width' in self.options else 100
        
        # Get displacement in Hz from original frequencies for each wave
        frequency_displacements = np.array([np.argmax(Sxx[f-flw:f+flw]) - flw for f in frequencies])
        
        # last = self.all_frequency_displacements[-1]
        # self.all_frequency_displacements.append(frequency_displacements)
        

        # not_outliers = ~DopplerAnalyzer.find_outliers(frequency_displacements)
        # frequency_displacements = frequency_displacements[not_outliers]
        # frequencies = frequencies[not_outliers]

        # if frequencies.size == 0:
        #     return 0

        # snr = np.array([np.max(Sxx[f-flw:f+flw]) for f in frequencies])
        
        # if self.options and 'noise_variance_weighted_mean' in self.options:
        
        # else:
        #     variances = None

        # if self.options and 'ignore_spikes' in self.options:
        #     difference = np.abs(self.all_frequency_displacements[-1] - frequency_displacements)
        #     greater_than_ten = difference > 10
        #     if np.all(greater_than_ten):
        #         frequency_displacements.fill(self.all_frequency_displacements[np.argmin(difference)])
        #     else: 
        #         frequency_displacements[greater_than_ten] = np.mean(frequency_displacements)

        frequency_displacements, frequencies = self.filter_frequencies(frequency_displacements, frequencies)

        if len(frequencies) == 0:
            return 0
        # Apply Doppler effect formula to compute speed in cm/s
        speeds = np.array([(frequency_displacements[i]/(frequency)) * 343.73 * 100 for i, frequency in enumerate(frequencies)])

        #variances = 1/variances
        mean = np.mean(speeds)#np.average(speeds, weights=snr)
        #TODO take into account that higher frequencies mean more speed
        
        
        return mean

    def filter_frequencies(self, frequency_displacements, frequencies):
        if not self.options:
            return frequency_displacements, frequencies

        if 'doppler_threshold' in self.options:
            try:
                threshold = self.options['doppler_threshold']['values'][self.options['doppler_threshold']['index']]
            except IndexError:
                print("error")
            threshold = self.options['doppler_threshold']['values'][self.options['doppler_threshold']['index']]
            thresholded_freqs = np.array(abs(frequency_displacements) <= threshold, dtype=bool)
            frequency_displacements = frequency_displacements[~thresholded_freqs]
            frequencies = frequencies[~thresholded_freqs]
        
        if len(frequencies) > 0 and 'outlier_removal' in self.options:
            max_deviation = self.options['outlier_removal']['values'][self.options['outlier_removal']['index']]
            not_outliers = ~DopplerAnalyzer.find_outliers(frequency_displacements, max_deviation=max_deviation)
            frequency_displacements = frequency_displacements[not_outliers]
            frequencies = frequencies[not_outliers]
        
        return frequency_displacements, frequencies
    
    @staticmethod
    def find_outliers(data, max_deviation=1.35): 
        return np.array(abs(data - np.median(data)) > max_deviation * np.std(data), dtype=bool)
