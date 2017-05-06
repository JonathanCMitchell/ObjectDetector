import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import time
# New
from Car_Detector import Car_Detector
df = pd.read_csv('./project_video_data/driving.csv')

cf = Car_Detector()
# I analyzed all the images in the dataset individually to evaluate threshold parameters
start = 0
stop = 1250
for i in tqdm(range(start, stop)):
    impath = df.iloc[[i]]['image_path'].values[0]
    img = mpimg.imread(impath)
    image, averaged_heatmaps, thresholded_heatmaps = cf.process_image(img)
    cv2.imwrite('./final_results/images/' + str(i) + 'image' + '.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.imwrite('./final_results/averaged_heatmaps/'+ str(i) + 'averaged_heatmap' + '.jpg', averaged_heatmaps)
    cv2.imwrite('./final_results/thresholded_heatmaps/'+ str(i) + 'thresholded_heatmap' + '.jpg', thresholded_heatmaps)

#
# This is used to create the movie
# from moviepy.editor import VideoFileClip
#
# t = time.time()
# test_output = 'project_video_output_car_nh1_sum.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# white_clip = clip1.fl_image(cf.process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(test_output, audio=False)
# t2 = time.time()
# print('time: ', (t2 - t) / 60)
