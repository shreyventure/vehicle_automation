import numpy as np
import cv2

def display_lane_roadway(image, fit_pts, M_wide_inv):
    """
    Colors the roadway of the vehicle's lane defined by the left and right fit points 
    """
    
    # Unpack and Define variables
    (left_fitx, right_fitx, ploty) = fit_pts    
    
    # Create an image to draw the lines on
    out_img = np.zeros_like(image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(out_img, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(out_img, M_wide_inv, (image.shape[1], image.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    return result


def display_corner_image(image, corner_image, scale_size = 1/3):
    """
    Displays an image at the top corner of another image.
    """
    
    (height, width) = image.shape[:2]
        
    # Create a black box at a corner of "image"
    box_startX, box_startY, box_endX, box_endY = int((1-scale_size)*width), 0, width, int(scale_size*height)  
    image[box_startY: box_endY, box_startX:box_endX] = 0    
    
    # Resize "corner_image"
    resized_width  = box_endX - box_startX
    resized_height = box_endY - box_startY 
    resized = cv2.resize(corner_image, (resized_width,resized_height), interpolation = cv2.INTER_AREA)

    # Combine the result with the original image
    image[box_startY: box_endY, box_startX:box_endX] = resized    
     
    return image