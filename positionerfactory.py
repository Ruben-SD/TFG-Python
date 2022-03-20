from predictor import Predictor, OfflinePredictor
from tracker import CameraTracker1D, CameraTracker2D, OfflineCameraTracker1D, OfflineCameraTracker2D


class PositionerFactory:
    @staticmethod
    def create_predictor(config, plotter):
        if config['offline']:
            return OfflinePredictor(config, plotter)
        else: 
            return Predictor(config, plotter)

    @staticmethod
    def create_tracker(config, plotter):
        if config['offline']:
            trackers_types = {"1D": OfflineCameraTracker1D,
                              "2D": OfflineCameraTracker2D}
            tracker = trackers_types[config['config']['tracker_type']]
            return tracker(config, plotter)
        else:
            trackers_types = {"1D": CameraTracker1D,
                              "2D": CameraTracker2D}
            tracker = trackers_types[config['tracker_type']]
            return tracker(config, plotter)
