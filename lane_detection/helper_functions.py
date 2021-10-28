import numpy as np
import cv2

def applyThresh(image, thresh=(0,255)):
    """
    Apply threshold to binary image. Setting to '1' pixels> minThresh & pixels <= maxThresh.
    """
    binary = np.zeros_like(image)
    binary[(image > thresh[0]) & (image <= thresh[1])] = 1
    return binary

def S_channel(image):
    """
    Returns the Saturation channel from an RGB image.
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    return S
    
def sobel_X(image):
    """
    Applies Sobel in the x direction to an RGB image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    abs_sobelx = np.abs(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3))
    sobelx     = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    return sobelx

def binary_pipeline(image):
    """
    Combination of color and gradient thresholds for lane detection. 
    Input image must be RGB
    """
    sobelx    = sobel_X(image)
    s_channel = S_channel(image)
    
    bin_sobelx    = applyThresh(sobelx, thresh=(20,100))
    bin_s_channel = applyThresh(s_channel, thresh=(90,255))
    
    return bin_sobelx | bin_s_channel

def find_lane_pixels_in_sliding_window(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    There is a left and right window sliding up independent from each other.
    This function returns the pixel coordinates contained within the sliding windows
    as well as the sliding windows midpoints
    PARAMETERS
    * nwindows : number of times window slides up
    * margin   : half of window's width  (+/- margin from center of window box)
    * minpix   : minimum number of pixels found to recenter window
    """
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    (height , width) = binary_warped.shape
    histogram = np.sum(binary_warped[int(height/2):,:], axis=0)
    window_leftx_midpoint  = np.argmax(histogram[:np.int(width/2)])
    window_rightx_midpoint = np.argmax(histogram[np.int(width/2):]) + np.int(width/2)

    # Set height of windows 
    window_height = np.int(height/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero  = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
  
    # Create empty lists 
    left_lane_inds  = []            # left lane pixel indices
    right_lane_inds = []            # Right lane pixel indices
    xleft_lane_win_midpts  = []     # left lane sliding window midpoints (x-coord)
    xright_lane_win_midpts = []     # Right lane sliding window midpoints (x-coord)
    
    # Step through the left and right windows one slide at a time
    for i in range(nwindows):
        # Identify right and left window boundaries 
        win_y_top       = height - (i+1)*window_height
        win_y_bottom    = height -   i  *window_height
        win_xleft_low   = max(window_leftx_midpoint  - margin , 0) 
        win_xleft_high  =     window_leftx_midpoint  + margin
        win_xright_low  =     window_rightx_midpoint - margin 
        win_xright_high = min(window_rightx_midpoint + margin , width)
        
        # Identify the nonzero pixels within the window and append to list
        good_left_inds  = ((nonzeroy >= win_y_top)      & (nonzeroy < win_y_bottom) & 
                           (nonzerox >= win_xleft_low)  & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_top)      & (nonzeroy < win_y_bottom) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.extend(good_left_inds)
        right_lane_inds.extend(good_right_inds)
        
        # Recenter next window midpoint If you found > minpix pixels and append previous midpoint
        xleft_lane_win_midpts.append(window_leftx_midpoint)
        xright_lane_win_midpts.append(window_rightx_midpoint)
        if len(good_left_inds  > minpix): window_leftx_midpoint  = np.mean(nonzerox[good_left_inds], dtype=np.int32)
        if len(good_right_inds > minpix): window_rightx_midpoint = np.mean(nonzerox[good_right_inds], dtype=np.int32)

    # Extract left and right line pixel positions
    xleft_lane  = nonzerox[left_lane_inds]
    yleft_lane  = nonzeroy[left_lane_inds] 
    xright_lane = nonzerox[right_lane_inds]
    yright_lane = nonzeroy[right_lane_inds]

    return (xleft_lane,yleft_lane), (xright_lane,yright_lane), (xleft_lane_win_midpts,xright_lane_win_midpts)


def draw_lane_pixels_in_sliding_window(binary_warped, left_lane_pts, right_lane_pts, window_midpts, margin=100):
    """
    Paints lane pixels and sliding windows.
    PARAMETERS
    * margin : half of window's width  (+/- margin from center of window box)
    """
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Unpack and Define variables
    (height , width) = binary_warped.shape
    (xleft_lane , yleft_lane)  = left_lane_pts
    (xright_lane, yright_lane) = right_lane_pts
    (xleft_lane_win_midpts, xright_lane_win_midpts) = window_midpts
    nwindows = len(xleft_lane_win_midpts)    # number of times window slided up
    window_height = int(height/nwindows)
    
    # Color left and right lane regions
    out_img[yleft_lane , xleft_lane]  = [255, 0, 0]
    out_img[yright_lane, xright_lane] = [0, 0, 255]
    
    # Draw the windows on the visualization image
    for i in range(nwindows):
        window_leftx_midpoint  = xleft_lane_win_midpts[i] 
        window_rightx_midpoint = xright_lane_win_midpts[i]
        win_y_top       = height - (i+1)*window_height
        win_y_bottom    = height -   i  *window_height
        win_xleft_low   = max(window_leftx_midpoint  - margin , 0) 
        win_xleft_high  =     window_leftx_midpoint  + margin
        win_xright_low  =     window_rightx_midpoint - margin 
        win_xright_high = min(window_rightx_midpoint + margin , width)
        
        cv2.rectangle(out_img,(win_xleft_low,win_y_top),
                              (win_xleft_high,win_y_bottom),(0,255,0), 12) 
        cv2.rectangle(out_img,(win_xright_low,win_y_top),
                              (win_xright_high,win_y_bottom),(0,255,0), 12) 
    return out_img


def ransac_polyfit(x, y, order=2, n=100, k=10, t=100, d=20, f=0.9):
    """
    RANSAC: finds and returns best model coefficients
    n – minimum number of data points required to fit the model
    k – maximum number of iterations allowed in the algorithm
    t – threshold value to determine when a data point fits a model
    d – number of close data points required to assert that a model fits well to data
    f – fraction of close data points required
    """
    besterr = np.inf
    bestfit = None
    if len(x) > 0:            #if input data not empty
        for kk in range(k):
            maybeinliers = np.random.randint(len(x), size=n)
            maybemodel = np.polyfit(x[maybeinliers], y[maybeinliers], order)
            alsoinliers = np.abs(np.polyval(maybemodel,x)-y) < t
            if sum(alsoinliers) > d and sum(alsoinliers) > len(x)*f:
                bettermodel = np.polyfit(x[alsoinliers], y[alsoinliers], order)
                thiserr = np.sum(np.abs(np.polyval(bettermodel,x[alsoinliers])-y[alsoinliers]))
                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr
    return bestfit

def fit_polynomial(img_height, left_lane_pts, right_lane_pts):
    """
    Returns pixel coordinates and polynomial coefficients of left and right lane fit.
    If empty lane pts are provided it returns coordinate (0,0) for left and right lane
    and sets fits to None.
    """
    # Unpack and Define variables 
    (xleft_lane , yleft_lane)  = left_lane_pts
    (xright_lane, yright_lane) = right_lane_pts

    try:
        # Fit a second order polynomial to each lane
        left_fit  = ransac_polyfit(yleft_lane , xleft_lane, order=2)
        right_fit = ransac_polyfit(yright_lane, xright_lane, order=2)
        #print(left_fit)
        #print(right_fit)
        
        # Generate x and y values of left and right fit
        ploty = np.linspace(0, img_height-1, img_height)
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)
    
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('[WARNING] The function failed to fit a line!')
        ploty      = 0
        left_fitx  = 0
        right_fitx = 0
        left_fit   = None
        right_fit  = None

    return left_fit, right_fit, left_fitx, right_fitx, ploty

def find_lane_pixels_around_poly(binary_warped, left_fit, right_fit, margin = 100):
    """
    Returns the pixel coordinates contained within a margin from left and right polynomial fits.
    Left and right fits shoud be from the previous frame.
    PARAMETER
    * margin: width around the polynomial fit
    """
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Search within the +/- margin of the polynomial from previous frame 
    left_lane_inds  = ((nonzerox >= (np.polyval(left_fit,nonzeroy)-margin)) &  (nonzerox <= (np.polyval(left_fit,nonzeroy)+margin))).nonzero()[0]
    right_lane_inds = ((nonzerox >= (np.polyval(right_fit,nonzeroy)-margin)) &  (nonzerox <= (np.polyval(right_fit,nonzeroy)+margin))).nonzero()[0]
    
    # Extract left and right line pixel positions    
    xleft_lane  = nonzerox[left_lane_inds]
    yleft_lane  = nonzeroy[left_lane_inds] 
    xright_lane = nonzerox[right_lane_inds]
    yright_lane = nonzeroy[right_lane_inds]
    
    return (xleft_lane,yleft_lane), (xright_lane,yright_lane)


def draw_lane_pixels_around_poly(binary_warped, left_lane_pts, right_lane_pts, previous_fit_pts, margin=100):
    """
    Paints lane pixels and poly fit margins. Poly fit margins are based on previous frame values.
    PARAMETER
    * margin: width around the polynomial fit
    """
    # Create two output images to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img_margins = np.zeros_like(out_img)
    
    # Unpack and Define variables
    (height , width) = binary_warped.shape
    (xleft_lane , yleft_lane)  = left_lane_pts
    (xright_lane, yright_lane) = right_lane_pts
    (left_fitx, right_fitx, ploty) = previous_fit_pts
    
    # Color left and right lane pixels
    out_img[yleft_lane , xleft_lane]  = [255, 0, 0]   # Red
    out_img[yright_lane, xright_lane] = [0, 0, 255]   # Blue
    
    # Color left and right previous polynomial fit. NOTE: type of fit values are returned in float
    for cx,cy in zip(np.int_(left_fitx), np.int_(ploty)):
        cv2.circle(out_img, (cx,cy), radius= 1, color=[255, 0, 255], thickness=10)
    for cx,cy in zip(np.int_(right_fitx), np.int_(ploty)):
        cv2.circle(out_img, (cx,cy), radius= 1, color=[255, 0, 255], thickness=10)    
                
    # Draw polynomial margins
    # Generate a polygon to illustrate the search area. NOTE: you flip array to keep contour when cv2.fillPoly
    left_line_left_margin  = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_right_margin = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))]) 
    left_line_margin_pts   = np.hstack((left_line_left_margin, left_line_right_margin))
    
    right_line_left_margin  = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_right_margin = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,ploty])))])
    right_line_margin_pts   = np.hstack((right_line_left_margin, right_line_right_margin))

    cv2.fillPoly(out_img_margins, np.int_([left_line_margin_pts]), (0,255, 0))
    cv2.fillPoly(out_img_margins, np.int_([right_line_margin_pts]), (0,255, 0))
    
    # Combine output images
    result = cv2.addWeighted(out_img, 1, out_img_margins, 0.3, 0)
    
    return result

def augment_previous_fit_pts(left_lane_pts, right_lane_pts, previous_fit_pts, density=4, line_width_margin=10):
    """
    Add to detected points the pts near previous line fits.
    NOTE: This function makes the points from the bottom half of previous line fits five times as dense.
    PARMETERS:
    * density           : number of times points are added near line fits.
    * line_width_margin : range of values generated near line fits
    """     
    # Unpack and Define variables
    (xleft_lane , yleft_lane)  = left_lane_pts
    (xright_lane, yright_lane) = right_lane_pts
    (left_fitx, right_fitx, ploty) = previous_fit_pts
    
    # Continue if there are points to add
    if len(ploty) > 1:
        
        # Create empty lists and array
        xleft_lane_aug  = []
        xright_lane_aug = []
        y_lane_aug      = np.array([])

        # Make previous line fits dense
        for i in range(density):
            xleft_lane_aug.extend([ x + np.random.randint(-line_width_margin, high=line_width_margin) for x in left_fitx])
            xright_lane_aug.extend([x + np.random.randint(-line_width_margin, high=line_width_margin) for x in right_fitx])
            y_lane_aug = np.hstack((y_lane_aug,ploty))
        
        # Make bottom half of previous line fits denser    
        bottom_pixel   = int(ploty[-1])
        midpoint_pixel = int(ploty[-1]/2)
        for i in range(5*density):
            xleft_lane_aug.extend([ x + np.random.randint(-line_width_margin, high=line_width_margin) for x in left_fitx[midpoint_pixel:bottom_pixel]])
            xright_lane_aug.extend([x + np.random.randint(-line_width_margin, high=line_width_margin) for x in right_fitx[midpoint_pixel:bottom_pixel]])
            y_lane_aug = np.hstack((y_lane_aug,ploty[midpoint_pixel:bottom_pixel]))
        
        # Augment
        xleft_lane_aug  = np.array(xleft_lane_aug)
        xright_lane_aug = np.array(xright_lane_aug)
        left_lane_pts_aug  = (np.hstack((xleft_lane , xleft_lane_aug)) , np.hstack((yleft_lane , y_lane_aug)))
        right_lane_pts_aug = (np.hstack((xright_lane, xright_lane_aug)), np.hstack((yright_lane, y_lane_aug)))

        return left_lane_pts_aug, right_lane_pts_aug