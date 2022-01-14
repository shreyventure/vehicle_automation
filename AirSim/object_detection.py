import airsim
import time
import cv2
import numpy as np
import keyboard
import os
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms

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
            img = img1d.reshape(response.height, response.width, 3)
            
            results = model(img)
            res = results.xyxy[0]
            labels = results.pandas().xyxy[0]['name']

            boxes = []
            for ele in res:
                boxes.append(list(ele[:4]))

            boxes = torch.tensor(boxes)

            img = transforms.ToTensor()(img.copy())
            img = img * 255
            img = img.type(torch.uint8)
            img = img.unsqueeze(0)

            drawn_boxes = draw_bounding_boxes(image=img[0], boxes= boxes, labels=results.pandas().xyxy[0]['name'], width=2, colors=['blue' for _ in range(len(labels))])
            
            tensor_to_pil = transforms.ToPILImage()(drawn_boxes.squeeze(0))
            pic = np.array(tensor_to_pil)
            
            cv2.imshow('detections', pic)
            cv2.waitKey(1)

        client.setCarControls(car_controls)

