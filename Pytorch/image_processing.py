import cv2
import numpy as np
import random

def add_random_shadow(img, w_low=0.6, w_high=0.85):
    """
    Overlays supplied image with a random shadow polygon
    The weight range (i.e. darkness) of the shadow can be configured via the interval [w_low, w_high)
    """
    if random.random() > 0.5:
      return img
    
    cols, rows = (img.shape[0], img.shape[1])
    
    top_y = np.random.random_sample() * rows
    bottom_y = np.random.random_sample() * rows
    bottom_y_right = bottom_y + np.random.random_sample() * (rows - bottom_y)
    top_y_right = top_y + np.random.random_sample() * (rows - top_y)
    if np.random.random_sample() <= 0.5:
        bottom_y_right = bottom_y - np.random.random_sample() * (bottom_y)
        top_y_right = top_y - np.random.random_sample() * (top_y)
    
    poly = np.asarray([[ [top_y,0], [bottom_y, cols], [bottom_y_right, cols], [top_y_right,0]]], dtype=np.int32)
        
    mask_weight = np.random.uniform(w_low, w_high)
    origin_weight = 1 - mask_weight
    
    mask = np.copy(img).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0))
    #masked_image = cv2.bitwise_and(img, mask)
    
    return cv2.addWeighted(img.astype(np.int32), origin_weight, mask, mask_weight, 0).astype(np.uint8)

def gaussian_noise(img):
  '''
  img: opencv image (numpy array)

  returns: a new opencv image (numpy array) with manipulated noise level.
  '''
  if random.random() > 0.5:
      return img
  mean = 0
  var = random.randint(50, 200)
  sigma = var ** 0.5
  gaussian = np.random.normal(mean, sigma, (img.shape[0],img.shape[1]))

  noisy_image = np.zeros(img.shape, np.float32)

  if len(img.shape) == 2:
      noisy_image = img + gaussian
  else:
      noisy_image[:, :, 0] = img[:, :, 0] + gaussian
      noisy_image[:, :, 1] = img[:, :, 1] + gaussian
      noisy_image[:, :, 2] = img[:, :, 2] + gaussian

  cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
  noisy_image = noisy_image.astype(np.uint8)
  
  return noisy_image
  
def increase_brightness(img):
    '''
    img: opencv image (numpy array)

    returns: a new opencv image (numpy array) with manipulated brightness level.
    '''
    if random.random() > 0.5:
      return img

    image = np.copy(img)
    new_image = np.zeros(image.shape, image.dtype)
    new_image = cv2.convertScaleAbs(image, alpha=1.20, beta=20)

    return new_image

def img_estim(img, thrshld):
    is_light = np.mean(img) > thrshld
    
    return img if is_light else increase_brightness(img)