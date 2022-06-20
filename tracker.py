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
        # self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
            self.look_smartphone_displacement_from_initial_pos()[:len(self.position)]
        return new_position

    def look_smartphone_displacement_from_initial_pos(self):
        _, frame = self.cam.read()
        
        cv2.waitKey(1)
        img_coords = self.get_smartphone_img_coords(frame)
        current_displacement = (img_coords - self.initial_smartphone_cam_pos) * self.cm_per_dim_pixel
        #print(img_coords, self.initial_smartphone_cam_pos, self.cm_per_dim_pixel)
        # (initial[1] + new_img_x / self.cm_per_dim_pixel_x)
        # np.array([(img_x - self.initial_smartphone_cam_pos[0]) * self.cm_per_length_pixel,
        #                             (img_y - self.initial_smartphone_cam_pos[1]) * self.cm_per_width_pixel])
        
        # Go from img to real space coords
        current_displacement *= -1
        if len(self.position) != 1:
            current_displacement = np.flip(current_displacement)
        return current_displacement

    @staticmethod
    def extract_smartphone_bounding_rect(frame):
        binary_img = CameraTracker.binarize_image(frame)
        # cv2.imshow("bin", cv2.bitwise_not(binary_img))
        # cv2.waitKey(1)
        improved_img = cv2.erode(cv2.bitwise_not(binary_img), np.ones((1, 4), dtype=int))
        # cv2.imshow("eroded", improved_img)
        # cv2.waitKey(1)
        pts_src = np.array([[511, 255], [294, 254], [181, 257], [58, 260], [491, 60], [292, 51], [188, 50], [77, 44], [521, 423], [294, 434], [172, 444], [50, 451]])

        pts_dst = np.array([[511, 255], [303.85, 255], [200.2, 255], [96.71, 255], [511, 37.27], [303.85, 37.27], [200.28, 37.27], [96.71, 37.27], [511, 418.29], [303.85, 418.29], [200.28, 418.29], [96.71, 418.29]])
        h, status = cv2.findHomography(pts_src, pts_dst)
        improved_img = cv2.warpPerspective(improved_img, h, (improved_img.shape[1], improved_img.shape[0]))
        cv2.imshow("Smartphone", improved_img)
        cv2.waitKey(1)
        contours, _ = cv2.findContours(improved_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        # index = -1
        # max_area = -1
        # max_area_index = -1
        # for i, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     if area > max_area:
        #         max_area = area
        #         max_area_index = i
        #     x, y, w, h = cv2.boundingRect(contour)

        #     if 6000 <= area <= 14000 and 1.5 <= w/h <= 4:
        #         index = i
        def disimilarity(contour):
            avg_phone_contour_area = 3500
            #normal_w_h_ratio = 
            contour_area = cv2.contourArea(contour)

            return abs(contour_area - avg_phone_contour_area)
        
        best_contour = sorted(contours, key = disimilarity)[0]
        
        return best_contour

    @staticmethod
    def get_smartphone_img_coords(frame):
        x, y, w, h = CameraTracker.extract_smartphone_bounding_rect(frame)
        
        return np.array([x, y])

    def stop(self):
        self.cam.release()


class OfflineCameraTracker(Positioner):
    def __init__(self, config, plotter):
        super().__init__('Tracker', config['config'], plotter)
        self.curr_frame = 0 # Start at 1 to skip initial pos
        coords_names = [data_name for data_name in config['data_names_to_plot'] if 'Tracker_position_' in data_name]
        self.camera_positions = [config[coords_name] for coords_name in coords_names]
        for x in self.camera_positions:
            x.append(x[-1])
        self.camera_positions = np.transpose(self.camera_positions)

    def update(self, dt):
        super().update(dt)
        current_position = self.obtain_current_position()
        self.set_position(current_position)

    def obtain_current_position(self):
        self.curr_frame += 1
        new_position = self.camera_positions[self.curr_frame]
        return new_position