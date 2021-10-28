import cv2
import numpy as np

def draw_roi_box(img, vertices, color=[0, 0, 255], thickness=5):
    """
    Draw a contour around region of interest on img (binary or color)
    Vertices must be 2D array of coordinate pairs [[(x1,y1),...,(x4,y4)]]
    
    """
    # Create blank image: if not color yet then create "color" binary image
    if len(img.shape) == 2:         #if single channel
        img = np.dstack((img, img, img))
    img_to_draw = np.copy(img)*0
    
    # Draw the lines
    for vertex in vertices:   # used 'for' to get rid of 2D array
        cv2.line(img_to_draw, tuple(vertex[0]), tuple(vertex[1]), color, thickness)
        cv2.line(img_to_draw, tuple(vertex[1]), tuple(vertex[2]), color, thickness)
        cv2.line(img_to_draw, tuple(vertex[2]), tuple(vertex[3]), color, thickness)
        cv2.line(img_to_draw, tuple(vertex[3]), tuple(vertex[0]), color, thickness)
           
    # Add detected lanes to original img
    output_img = cv2.addWeighted(img_to_draw, 1, img, 0.4, 0) 
    return output_img