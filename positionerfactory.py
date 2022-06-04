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
        if config['positioning'].get('tracker', False):
            positioners.append(PositionerFactory.create_tracker(config, plotter))
        return positioners