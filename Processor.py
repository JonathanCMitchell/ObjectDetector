# import LaneFinder from './LaneLineDetection/laneFinder.py'
import pickle as pickle
import settings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from VehicleDetection.helpers import draw_centroids

# Import Lane Line Finder dependencies
data = pickle.load( open( "./LaneLineDetection/camera_calibration.p", "rb" ) )
camera_matrix = data['mtx']
dist_coeffs = data['dist']

perspective_transform_data = pickle.load(open("./LaneLineDetection/perspective.p", 'rb'))
x_pixels_per_meter = perspective_transform_data['x_pixels_per_meter']
y_pixels_per_meter = perspective_transform_data['y_pixels_per_meter']
M = perspective_transform_data['homography_matrix']
src_pts = perspective_transform_data['source_points']

# Import Vehicle Detection dependencies
from VehicleDetection.Car_Detector import Car_Detector



from LaneLineDetection.laneFinder import LaneFinder
class Processor():
    def __init__(self):
        self.lanefinder = LaneFinder(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, camera_matrix, dist_coeffs,
                        M, x_pixels_per_meter, y_pixels_per_meter)
        self.vehicle_detector = Car_Detector()
        self.out_img = np.zeros((settings.IMG_HEIGHT, settings.IMG_WIDTH, 3), dtype = np.uint8)
    def process_image(self, input_image):
        lane_image = self.lanefinder.process_image(input_image)
        centroids = self.vehicle_detector.process_centroids(input_image)
        drawn_img = draw_centroids(lane_image, centroids)
        return drawn_img


p = Processor

from moviepy.editor import VideoFileClip

test_output = 'project_video_output_both.mp4'
clip1 = VideoFileClip("project_video.mp4")
p = Processor()
white_clip = clip1.fl_image(p.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(test_output, audio=False)
