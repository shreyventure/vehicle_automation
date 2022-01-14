import airsim
import time
import cv2
import numpy as np
import keyboard
import os
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

'''
Scene = 0, 
DepthPlanar = 1, 
DepthPerspective = 2,
DepthVis = 3, 
DisparityNormalized = 4,
Segmentation = 5,
SurfaceNormals = 6,
Infrared = 7,
OpticalFlow = 8,
OpticalFlowVis = 9
'''

client = airsim.CarClient() # https://microsoft.github.io/AirSim/api_docs/html/_modules/airsim/client.html#CarClient
client.confirmConnection()
apiControl = False
client.enableApiControl(apiControl)
client.armDisarm(apiControl)
car_controls = airsim.CarControls()

while True:

    responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
    for response in responses:
        if response.pixels_as_float:
            print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
            airsim.write_pfm('py1.pfm', airsim.get_pfm_array(response))
        else:
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            cv2.imwrite('new.png', img_rgb)

            results = model(['new.png'])

            results.print()
            results.save()
            det = cv2.imread('runs/detect/exp/new.jpg')
            cv2.imshow('detected', det)
            cv2.waitKey(1)

            os.remove('runs/detect/exp/new.jpg')
            os.rmdir('runs/detect/exp')

        client.setCarControls(car_controls)

