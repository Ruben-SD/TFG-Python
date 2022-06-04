from predictor import Predictor, OfflinePredictor
from tracker import CameraTracker1D, CameraTracker2D, OfflineCameraTracker1D, OfflineCameraTracker2D


class PositionerFactory:
    @staticmethod
    def create_predictor(config, plotter):
        if config.get('offline', False):
            return OfflinePredictor(config, plotter)
        else: 
            return Predictor(config, plotter)

    @staticmethod
    def create_tracker(config, plotter):
        tracker_type = config['positioning']['type']
        if config.get('offline', False):
            trackers = {"1D": OfflineCameraTracker1D,
                              "2D": OfflineCameraTracker2D}
            tracker = trackers[tracker_type]
            return tracker(config, plotter)
        else:
            trackers = {"1D": CameraTracker1D,
                              "2D": CameraTracker2D}
            tracker = trackers[tracker_type]
            return tracker(config, plotter)

    @staticmethod
    def create_positioners(config, plotter):
        positioners = [PositionerFactory.create_predictor(config, plotter)]
        if config['positioning'].get('tracker', False):
            positioners.append(PositionerFactory.create_tracker(config, plotter))
        return positioners