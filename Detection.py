import pickle as pickle
import pandas as pd
import settings
from laneFinder import LaneFinder
data = pickle.load( open( "camera_calibration.p", "rb" ) )
camera_matrix = data['mtx']
dist_coeffs = data['dist']


perspective_transform_data = pickle.load(open("perspective.p", 'rb'))
x_pixels_per_meter = perspective_transform_data['x_pixels_per_meter']
y_pixels_per_meter = perspective_transform_data['y_pixels_per_meter']
M = perspective_transform_data['homography_matrix']
src_pts = perspective_transform_data['source_points']

# read in dataframe
df = pd.read_csv('./data/driving.csv')

lf = LaneFinder(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, camera_matrix, dist_coeffs,
                        M, x_pixels_per_meter, y_pixels_per_meter)

from moviepy.editor import VideoFileClip

test_output = 'project_video_output_both.mp4'
clip1 = VideoFileClip("project_video.mp4")
lf = LaneFinder(settings.ORIGINAL_SIZE, settings.UNWARPED_SIZE, camera_matrix, dist_coeffs,
                        M, x_pixels_per_meter, y_pixels_per_meter)
white_clip = clip1.fl_image(lf.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(test_output, audio=False)
