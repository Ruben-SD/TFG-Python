from predictor import Predictor
from tracker import CameraTracker1D, CameraTracker2D


class PositionerFactory:
    @staticmethod
    def create_predictor(config):
        return Predictor(config)

    @staticmethod
    def create_tracker(config):
        trackers_types = {"1D": CameraTracker1D,
                          "2D": CameraTracker2D}
        tracker = trackers_types[config['tracker_type']]
        return tracker(config)
