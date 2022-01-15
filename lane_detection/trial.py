import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from lane_detection.calibrate_camera import calibrate_camera
from lane_detection.radius_of_curvature import measure_curvature_meters
from lane_detection.lane_centering import lane_center_deviation_meters 
from lane_detection.helper_functions import applyThresh, augment_previous_fit_pts, binary_pipeline, draw_lane_pixels_around_poly, draw_lane_pixels_in_sliding_window, find_lane_pixels_around_poly, find_lane_pixels_in_sliding_window, fit_polynomial, sobel_X, S_channel

print("Calibrating camera...")
ret, mtx, dist, rvec, tvec = calibrate_camera(directory="camera_calibration_images")

image = cv2.imread("road.jpg") #first image from list
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720
print('\n Location  :', "road.jpg", '\n Dimensions:', image.shape)
image = image[60:-25, :, :]
image = cv2.resize(image, (1280, 720), cv2.INTER_AREA)
image = cv2.undistort(image, mtx, dist, None, mtx)

# Calculate area of interest manually
height, width = image.shape[:2]
offset  = 50   # camera offset from center of car
bottom_left   =( 0   , height) 
upper_left    =( width/2 - width/8, 100)
upper_right   =( width/2 + width/8, 100)
bottom_right  =( width-140, height)
vertices = np.array([[bottom_left,upper_left,upper_right,bottom_right]], dtype=np.int32)

src = np.float32(vertices[0]) # with [0] we reduce one dimension

# Destination Points   
dst_wide = np.float32([bottom_left,
                   [bottom_left[0], 0],
                   [bottom_right[0],0],
                   bottom_right])
dst_narrow = np.float32([[upper_left[0],height],
                         upper_left,
                         upper_right,
                         [upper_right[0],height]])

# Perspective Transform Matrix
M_wide   = cv2.getPerspectiveTransform(src, dst_wide)
M_narrow = cv2.getPerspectiveTransform(src, dst_narrow)
M_wide_inv   = cv2.getPerspectiveTransform(dst_wide,src)    # Inverse Matrices
M_narrow_inv = cv2.getPerspectiveTransform(dst_narrow,src)

# Warped Images
warped_wide   = cv2.warpPerspective(image, M_wide, (width,height))
warped_narrow = cv2.warpPerspective(image, M_narrow, (width,height))

# Binary pipeline
bin_sobelx    = applyThresh(sobel_X(image), thresh=(20,100))
bin_s_channel = applyThresh(S_channel(image), thresh=(90,255))
bin_image     = binary_pipeline(image)
bin_warped_wide   = cv2.warpPerspective(bin_image, M_wide, (width,height))
bin_warped_narrow = cv2.warpPerspective(bin_image, M_narrow, (width,height))

# Fit Line Through Sliding Window Technique
left_lane_pts, right_lane_pts, window_midpts = find_lane_pixels_in_sliding_window(bin_warped_wide)
img_out = draw_lane_pixels_in_sliding_window(bin_warped_wide, left_lane_pts, right_lane_pts, window_midpts)

# Get line fit
left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(height, left_lane_pts, right_lane_pts)

tx = 50      # translate in x
ty = 100     # translate in y
M  = np.float32([[1,0,tx],[0,1,ty]])
next_frame = cv2.warpAffine(bin_warped_wide, M, (width,height))     

# Find pixels and Draw (Around Previous lane fit)
left_lane_pts, right_lane_pts = find_lane_pixels_around_poly(next_frame, left_fit, right_fit)
previous_fit_pts = (left_fitx, right_fitx, ploty)
img_out = draw_lane_pixels_around_poly(next_frame, left_lane_pts, right_lane_pts, previous_fit_pts)

# Get Line Fit
left_lane_pts_aug, right_lane_pts_aug = augment_previous_fit_pts(left_lane_pts, right_lane_pts, previous_fit_pts)
new_left_fit, new_right_fit, new_left_fitx, new_right_fitx, ploty = fit_polynomial(height, left_lane_pts_aug, right_lane_pts_aug)

left_radius_curve, right_radius_curve = measure_curvature_meters(height, previous_fit_pts)
print("\n Left R: {0:.2f} m \t Right R: {1:.2f} m ".format(left_radius_curve, right_radius_curve))

# Calculate the radius of curvature at the bottom of image(height) in meters for both lane lines
lane_deviation = lane_center_deviation_meters(width,previous_fit_pts)
print("\n Lane Deviation: {0:.2f} m ".format(lane_deviation))