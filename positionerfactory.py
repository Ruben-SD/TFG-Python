from predictor import Predictor, OfflinePredictor
from tracker import CameraTracker1D, CameraTracker2D, OfflineCameraTracker


class PositionerFactory:
    @staticmethod
    def create_predictor(config):
        if config['offline']:
            return OfflinePredictor(config)
        else: 
            return Predictor(config)

    @staticmethod
    def create_tracker(config):
        if config['offline']:
            return OfflineCameraTracker(config)
        else:
            trackers_types = {"1D": CameraTracker1D,
                              "2D": CameraTracker2D}
            tracker = trackers_types[config['tracker_type']]
            return tracker(config)
