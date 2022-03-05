import cv2
import numpy as np
from positioner import Position2D, Positioner
from plotter import *
import copy

class CameraTracker(Positioner):
    def __init__(self, config):
        if not isinstance(self, CameraTracker2D):
            super().__init__(config)
        self.init_camera()
        _, first_frame = self.cam.read()   
        self.init_smartphone_data(config, first_frame)    
              
    def init_camera(self):
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
    
    def init_smartphone_data(self, config, frame):
        self.smartphone_dims = config['smartphone']['dims']
        _, _, length, width = CameraTracker.extract_smartphone_bounding_rect(frame)
        self.cm_per_length_pixel = self.smartphone_dims['length']/length
        self.cm_per_width_pixel = self.smartphone_dims['width']/width
        self.initial_smartphone_cam_pos = self.get_smartphone_img_coords(frame) 
        self.initial_position = copy.deepcopy(self.position)

    def update(self, dt):
        new_position = self.obtain_current_position()
        self.position.set(new_position)
        # if self.two_dimensions:
        #     self.set_position_data(self.initial_distance - distance_from_initial_pos)
        #     plotter.add_sample('real_y_position', self.get_distance()[1])
        # else: 
        #     self.set_position_data(self.initial_distance - distance_from_initial_pos[0])
        # plotter.add_sample('real_x_position', self.get_distance()[0])        

    def obtain_current_position(self):
        new_position = self.initial_position + self.look_smartphone_distance_from_initial_pos()
        return new_position

    def look_smartphone_distance_from_initial_pos(self):
        raise NotImplemented()

    @staticmethod
    def extract_smartphone_bounding_rect(frame):
        binary_img = CameraTracker.binarize_image(frame)
        improved_img = cv2.erode(binary_img, np.ones(12, dtype=int))
        contours, _ = cv2.findContours(improved_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        smartphone_contour = CameraTracker.find_smartphone_contour(contours, frame)
        x, y, w, h = cv2.boundingRect(smartphone_contour)
        return (x, y, w, h)

    @staticmethod
    def binarize_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
        _, binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod #TODO Could take into consideration smartphone dimensions to improve detection
    def find_smartphone_contour(contours, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        maxArea = -1
        index = -1
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            mean = cv2.mean(gray_img[y:y+h, x:x+w])[0]
            area = cv2.contourArea(contour)
            if area > maxArea and mean < 80:
                index = i
                maxArea = area

        if index == -1:
            raise ValueError("Cannot find smartphone shaped black contour in image")
        return contours[index]

    @staticmethod
    def get_smartphone_img_coords(frame):
        x, y, w, h = CameraTracker.extract_smartphone_bounding_rect(frame)
        x = int(x + w/2)
        y = int(y + h/2)
        return (x, y)

    def __del__(self):
        self.cam.release()

class CameraTracker1D(CameraTracker):
    def __init__(self, config):
        super().__init__(config)

    def look_smartphone_distance_from_initial_pos(self):
        _, frame = self.cam.read()
        img_x, _ = self.get_smartphone_img_coords(frame)  
        current_distance = np.array([(self.initial_smartphone_cam_pos[0] - img_x) * self.cm_per_length_pixel])
        return current_distance

    
class CameraTracker2D(CameraTracker):
    def __init__(self, config):
        self.position = Position2D(config)
        super().__init__(config)        
        
        #position_configs = config['smartphone']['position'] 
        #self.initial_measuring_tape_position = np.array([[position for _, position in position_config.items()] for position_config in position_configs], dtype=float)
        # if self.two_dimensions:
        #     plotter.add_data('real_y_position', [], plot=True)    
        # plotter.add_data('real_x_position', [], plot=True)

    def look_smartphone_distance_from_initial_pos(self):
        _, frame = self.cam.read()
        (img_x, img_y) = self.get_smartphone_img_coords(frame)  
        current_distance = np.array([(img_x - self.initial_smartphone_cam_pos[0]) * self.cm_per_length_pixel, (img_y - self.initial_smartphone_cam_pos[1]) * self.cm_per_width_pixel])
        return current_distance