import numpy as np

def measure_curvature_pixels(y_eval, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels.
    PARAMETERS
    * y_eval : where we want radius of curvature to be evaluated (We'll choose the maximum y-value, bottom of image)
    '''
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad


def measure_curvature_meters(y_eval, fit_pts, ym_per_pix= 30/720, xm_per_pix=3.7/900):
    '''
    Calculates the curvature of polynomial functions in meters. 
    NOTE: Chose a straight line birdeye view image to calculate pixel to meter parameters.
    PARAMETERS
    * ym_per_pix : meters per pixel in y dimension (meters/length of lane in pixel)
    * xm_per_pix : meters per pixel in x dimension (meters/width between lanes in pixel)
    * y_eval     : where we want radius of curvature to be evaluated (We'll choose the maximum y-value, bottom of image)
    '''
    # Unpack and Define variables
    (left_fitx, right_fitx, ploty) = fit_pts
    
    # Fit new polynomials to x,y in world space
    left_fit_cr  = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
       
    # Calculation of R_curve (radius of curvature)
    left_radius_curve  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_radius_curve = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_radius_curve, right_radius_curve