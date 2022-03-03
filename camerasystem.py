from operator import length_hint
import cv2
import numpy as np
from positioner import Positioner
from plotter import *

class CameraSystem(Positioner):
    def __init__(self, config):
        super().__init__(config)
        self.SMARTPHONE_LENGTH_CM = config['smartphone']['dims']['length']
        self.SMARTPHONE_WIDTH_CM = config['smartphone']['dims']['width']

        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0) 

        _, first_frame = self.cam.read()
        self.cm_per_dims_px = self.get_cm_per_smartphone_px_width(first_frame)
        self.initial_smartphone_cam_pos = self.get_smartphone_img_coords(first_frame)
        
        #position_configs = config['smartphone']['position'] 
        #self.initial_measuring_tape_position = np.array([[position for _, position in position_config.items()] for position_config in position_configs], dtype=float)
        if self.two_dimensions:
            plotter.add_data('real_y_position', [], plot=True)    
        plotter.add_data('real_x_position', [], plot=True)

    def update_position(self, dt):
        distance_from_initial_pos = self.get_smartphone_distance_from_initial_pos()
        if self.two_dimensions:
            self.set_distance(self.initial_distance - distance_from_initial_pos)
            plotter.add_sample('real_y_position', self.get_distance()[1])
        else: 
            self.set_distance(self.initial_distance - distance_from_initial_pos[0])
        plotter.add_sample('real_x_position', self.get_distance()[0])        
        return self.get_position()

    def get_smartphone_distance_from_initial_pos(self):
        _, frame = self.cam.read()
        (imgX, imgY) = self.get_smartphone_img_coords(frame)  
        current_distance = (imgX - self.initial_smartphone_cam_pos[0]) * self.cm_per_dims_px[0], (imgY - self.initial_smartphone_cam_pos[1]) * self.cm_per_dims_px[1]
        return current_distance

    def get_smartphone_img_coords(self, frame):
        x, y, w, h = self.get_smartphone_bounding_rect(frame)
        x = int(x + w/2)
        y = int(y + h/2)
        return (x, y)

    def get_smartphone_dims(self, frame):
        _, _, w, h = self.get_smartphone_bounding_rect(frame)
        return (w, h)

    def get_smartphone_bounding_rect(self, frame):
        binary = self.binarize_image(frame)
        improved = self.improve_binary_img(binary)
        contours, _ = cv2.findContours(improved, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        smartphone_contour = self.get_smartphone_contour(contours, frame)
        x, y, w, h = cv2.boundingRect(smartphone_contour)
        return (x, y, w, h)

    def binarize_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
        _, binary = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
        return binary

    def improve_binary_img(self, binary):
        eroded = cv2.erode(binary, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1]))
        return eroded

    @staticmethod #TODO Could take into consideration smartphone dimensions to improve detection
    def get_smartphone_contour(contours, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
        maxArea = -1
        index = -1
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            mean = cv2.mean(gray[y:y+h, x:x+w])[0]
            area = cv2.contourArea(contour)
            if area > maxArea and mean < 80:
                index = i
                maxArea = area

        if index == -1:
            raise ValueError("Cannot find smartphone shaped black contour in image")
        return contours[index]

    def get_cm_per_smartphone_px_width(self, img):
        length, width = self.get_smartphone_dims(img)
        return self.SMARTPHONE_LENGTH_CM/length, self.SMARTPHONE_WIDTH_CM/width

    def __del__(self):
        self.cam.release()