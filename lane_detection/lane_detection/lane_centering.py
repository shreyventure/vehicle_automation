import numpy as np
def lane_center_deviation_meters(img_width,fit_pts, xm_per_pix=3.7/900):
    '''
    Calculates the deviation from the center of the lanes in meters. 
    NOTE: Chose a straight line birdeye view image to calculate pixel to meter parameters.
    PARAMETERS
    * xm_per_pix : meters per pixel in x dimension (meters/width between lanes in pixel)
    '''
    # Unpack and Define variables
    (left_fitx, right_fitx, ploty) = fit_pts
    lane_center  = int((left_fitx[-1] + right_fitx[-1])/2) # choose last (bottom of image) pixels
    image_center = int(img_width/2)
    
    # Calculation of lane center deviation in meters
    deviation = np.abs(image_center-lane_center)* xm_per_pix

    return deviation