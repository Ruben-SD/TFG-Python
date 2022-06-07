from predictor import Predictor, OfflinePredictor
from tracker import CameraTracker, OfflineCameraTracker

class PositionerFactory:
    @staticmethod
    def create_predictor(config, plotter):
        if config.get('offline', False):
            return OfflinePredictor(config, plotter)
        else: 
            return Predictor(config, plotter)

    @staticmethod
    def create_tracker(config, plotter):
        if config.get('offline', False):
            return OfflineCameraTracker(config, plotter)
        else:
            return CameraTracker(config, plotter)

    @staticmethod
    def create_positioners(config, plotter):
        positioners = [PositionerFactory.create_predictor(config, plotter)]
        enable_tracker = config['config']['positioning']['tracker'] if 'config' in config else config['positioning']['tracker']
        if enable_tracker:
            positioners.append(PositionerFactory.create_tracker(config, plotter))
        return positioners