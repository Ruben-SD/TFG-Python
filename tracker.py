import cv2
import numpy as np
from positioner import Positioner
import copy


class CameraTracker(Positioner):
    def __init__(self, config, plotter):
        super().__init__('Tracker', config, plotter)
        self.initial_position = self.position
        self.init_camera()
        _, first_frame = self.cam.read()
        self.init_smartphone_data(config, first_frame)

    def init_camera(self):
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    def init_smartphone_data(self, config, frame):
        self.smartphone_dims = config['smartphone']['dims']
        _, _, length, width = CameraTracker.extract_smartphone_bounding_rect(
            frame)
        self.cm_per_dim_pixel = np.array([self.smartphone_dims['length']/length, self.smartphone_dims['width']/width])
        self.initial_smartphone_cam_pos = self.get_smartphone_img_coords(frame)
        self.initial_position = copy.deepcopy(self.position)

    def update(self, dt):
        super().update(dt)
        current_position = self.obtain_current_position()
        self.set_position(current_position)
        # if self.two_dimensions:
        #     self.set_position_data(self.initial_distance - distance_from_initial_pos)
        #     plotter.add_sample('real_y_position', self.get_distance()[1])
        # else:
        #     self.set_position_data(self.initial_distance - distance_from_initial_pos[0])
        # plotter.add_sample('real_x_position', self.get_distance()[0])

    def obtain_current_position(self):
        new_position = self.initial_position + \
            self.look_smartphone_displacement_from_initial_pos()
        return new_position

    def look_smartphone_displacement_from_initial_pos(self):
        _, frame = self.cam.read()
        cv2.imshow("Smartphone", frame)
        cv2.waitKey(1)
        img_coords = self.get_smartphone_img_coords(frame)
        current_displacement = (img_coords - self.initial_smartphone_cam_pos) * self.cm_per_dim_pixel
        # np.array([(img_x - self.initial_smartphone_cam_pos[0]) * self.cm_per_length_pixel,
        #                             (img_y - self.initial_smartphone_cam_pos[1]) * self.cm_per_width_pixel])
        
        # Go from img to real space coords
        current_displacement[0] *= -1
        current_displacement = np.flip(current_displacement)
        return current_displacement

    @staticmethod
    def extract_smartphone_bounding_rect(frame):
        binary_img = CameraTracker.binarize_image(frame)
        improved_img = cv2.erode(binary_img, np.ones(12, dtype=int))
        contours, _ = cv2.findContours(
            improved_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        smartphone_contour = CameraTracker.find_smartphone_contour(contours)

        x, y, w, h = cv2.boundingRect(smartphone_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
        cv2.imshow("Smartphone", frame)
        cv2.waitKey(1)
        return (x, y, w, h)

    @staticmethod
    def binarize_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod  # TODO Could take into consideration smartphone dimensions to improve detection
    def find_smartphone_contour(contours):
        index = -1
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if 6000 <= area <= 14000:
                index = i

        # if index == -1:
        #     import winsound, time
        #     winsound.Beep(5000, 2)
        #     time.sleep(2)
        #     raise ValueError("Cannot find smartphone shaped black contour in image")
        return contours[index]

    @staticmethod
    def get_smartphone_img_coords(frame):
        x, y, w, h = CameraTracker.extract_smartphone_bounding_rect(frame)
        return np.array([x, y])

    def stop(self):
        self.cam.release()


class OfflineCameraTracker(Positioner):
    def __init__(self, config, plotter):
        super().__init__('Tracker', config['config'], plotter)
        self.curr_frame = -1
        coords_names = [data_name for data_name in config['data_names_to_plot'] if 'Tracker_position_' in data_name]
        self.camera_positions = np.transpose([config[coords_name] for coords_name in coords_names])

    def update(self, dt):
        super().update(dt)
        current_position = self.obtain_current_position()
        self.set_position(current_position)

    def obtain_current_position(self):
        self.curr_frame += 1
        new_position = self.camera_positions[self.curr_frame]
        return new_position