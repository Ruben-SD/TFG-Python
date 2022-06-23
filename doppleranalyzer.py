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

        self.kalman_filter = KalmanFilter(dim_x=2, dim_z=1)
        self.kalman_filter.x = np.array([[0.],
                [0.]])       # initial state (location and velocity)

        self.kalman_filter.F = np.array([[1.,1.],
                        [0.,1.]])    # state transition matrix

        self.kalman_filter.H = np.array([[0.,1.]])    # Measurement function
        self.kalman_filter.P *= 10                 # covariance matrix
        self.kalman_filter.R = 0.001                      # state uncertainty
        self.kalman_filter.Q = Q_discrete_white_noise(2, 1/24.0, .1) # process uncertainty

    def extract_speeds_from(self, audio_samples):
        _, _, Sxx = signal.spectrogram(audio_samples, fs=44100, nfft=44100, nperseg=len(audio_samples), mode='magnitude')
        
        speed = self.extract_speed_from(Sxx, np.array(self.frequencies))
        
        self.plotter.add_sample(f'Doppler_deviation_filtered_{self.id}', speed)

        self.kalman_filter.predict()
        self.kalman_filter.update(speed)

        # self.plotter.add_sample(f'Doppler_deviation_filtered_{self.id}_K', self.my_filter.x[1][0])
        if self.options is not None and 'kalman_filter' in self.options:
            return self.kalman_filter.x[1][0]
        
        return speed

    def extract_speed_from(self, Sxx, frequencies):
        # Frequency lookup width
        flw = self.options['frequency_lookup_width']["values"][self.options['frequency_lookup_width']["index"]] if self.options is not None and 'frequency_lookup_width' in self.options else 100
        
        # Get displacement in Hz from original frequencies for each wave
        frequency_displacements = np.array([np.argmax(Sxx[f-flw:f+flw]) - flw for f in frequencies])

        for f, disp in zip(frequencies, frequency_displacements):
            speed = disp/f * 343.73 * 100
            self.plotter.add_sample(f'Doppler_deviation_{f}_Hz', speed)

        # last = self.all_frequency_displacements[-1]
        # self.all_frequency_displacements.append(frequency_displacements)
        

        # not_outliers = ~DopplerAnalyzer.find_outliers(frequency_displacements)
        # frequency_displacements = frequency_displacements[not_outliers]
        # frequencies = frequencies[not_outliers]

        # if frequencies.size == 0:
        #     return 0

        snr = np.array([np.max(Sxx[f-flw:f+flw]) for f in frequencies])
        
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

        frequency_displacements, frequencies, snr = self.filter_frequencies(frequency_displacements, frequencies, snr)

        if len(frequencies) == 0:
            return 0
        # Apply Doppler effect formula to compute speed in cm/s
        speeds = np.array([(frequency_displacements[i]/(frequency)) * 343.73 * 100 for i, frequency in enumerate(frequencies)])

        #variances = 1/variances
        if 'snr_avg' in self.options:
            mean = np.average(speeds, weights=snr)
        else:
            mean = np.mean(speeds)#
        #TODO take into account that higher frequencies mean more speed
        
        return mean

    def filter_frequencies(self, frequency_displacements, frequencies, snr):
        # if not self.options:
        #     return frequency_displacements, frequencies

        if 'outlier_removal' in self.options:
            max_deviation = self.options['outlier_removal']['values'][self.options['outlier_removal']['index']]
            not_outliers = ~DopplerAnalyzer.find_outliers(frequency_displacements, max_deviation=max_deviation)
            frequency_displacements = frequency_displacements[not_outliers]
            frequencies = frequencies[not_outliers]
            snr = snr[not_outliers]

        if 'doppler_threshold' in self.options:
            threshold = self.options['doppler_threshold']['values'][self.options['doppler_threshold']['index']]
            thresholded_freqs = np.array(np.abs(frequency_displacements) <= threshold, dtype=bool)
            frequency_displacements = frequency_displacements[~thresholded_freqs]
            frequencies = frequencies[~thresholded_freqs]
            snr = snr[~thresholded_freqs]
        
        return frequency_displacements, frequencies, snr
    
    @staticmethod
    def find_outliers(data, max_deviation=1.35): 
        return np.array(abs(data - np.median(data)) > max_deviation * np.std(data), dtype=bool)
