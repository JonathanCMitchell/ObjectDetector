import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle
import cv2
import glob
import settings
import collections
from scipy.ndimage.measurements import label
from helpers import convert_color, \
    get_hog_features, \
    bin_spatial, \
    color_hist, \
    draw_centroids, \
    draw_labeled_boxes, \
    find_cars

import pandas as pd
from tqdm import tqdm

pickle_data = pickle.load(open("svc_pickle.p", "rb"))
svc = pickle_data['svc']
X_scaler = pickle_data['X_scaler']
orient = pickle_data['orient']
pix_per_cell = pickle_data['pix_per_cell']
cell_per_block = pickle_data['cell_per_block']
spatial_size = pickle_data['spatial_size']
hist_bins = pickle_data['hist_bins']


class Car_Detector():
    def __init__(self):
        self.threshold = 0.58
        self.smooth_factor = 8
        self.heatmaps = collections.deque(maxlen=10)
        self.count = 0
        self.frames_missed = 0
        self.found = None
        self.heat = np.zeros((settings.IMG_HEIGHT, settings.IMG_WIDTH), dtype=np.float32)  # maybe chance dtype

    def process_image(self, img):
        self.count += 1
        centroid_rectangles, average_heatmaps, thresholded_heatmaps = self.get_centroid_rectangles(img)
        draw_img = draw_centroids(img, centroid_rectangles)
        return draw_img, average_heatmaps, thresholded_heatmaps

        # return draw_img

    def get_detections(self, img):
        detection_general = []
        # Use multiple windows

        ystart = 400
        ystop = 656
        scale = 1.5
        centroid_rectangles = []
        detection_general.append(find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                           cell_per_block,
                                           spatial_size,
                                           hist_bins))

        ystart = 432
        ystop = 580 # prev is 560
        scale = 2
        detection_general.append(find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                           cell_per_block - 1,
                                           spatial_size,
                                           hist_bins))

        ystart = 400
        ystop = 560
        scale = 2.5
        detection_general.append(find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                                           cell_per_block,
                                           spatial_size,
                                           hist_bins))

        detections = [detection for detections in detection_general for detection in detections]
        if len(detections) == 0:
            self.frames_missed += 1
        elif len(detections) > 0:
            self.frames_missed = 0
        return detections

    def get_centroid_rectangles(self, img):
        centroid_rectangles = []

        detections = self.get_detections(img)
        heat = np.zeros((settings.IMG_HEIGHT, settings.IMG_WIDTH), dtype=np.float32)  # maybe chance dtype
        heat = self.update_heatmap(detections, heat)
        self.heatmaps.append(heat)

        thresholded_heatmap = self.apply_threshold(heat.astype(np.uint8), self.threshold)

        _, contours, hier = cv2.findContours(thresholded_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rect = cv2.boundingRect(contour)
            if rect[2] < 50 or rect[3] < 50: continue
            x, y, w, h = rect
            centroid_rectangles.append([x, y, x + w, y + h])
        # Now heatmap is binary so we apply contours
        return centroid_rectangles, np.mean(self.heatmaps, axis = 0), thresholded_heatmap

    def update_heatmap(self, detections, heat):
        for (x1, y1, x2, y2) in detections:
            heat[y1:y2, x1:x2] += 1
        return heat

    def apply_threshold(self, heatmap, threshold):
        if len(self.heatmaps) > self.smooth_factor:
            heatmap = np.sum(self.heatmaps, axis=0).astype(np.uint8)
            threshold = self.threshold * len(self.heatmaps) # added 0.5
        else:
            heatmap = heatmap
            # Make threshold a constant before averaging to rule out false detections
            threshold = 1

        # Threshold
        _, binary = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
        return binary
