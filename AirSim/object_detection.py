import airsim
import cv2
import numpy as np

import torch
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
from torch.autograd import Variable
import torchvision.transforms as transforms
import keyboard, time
from model import *

print('[INFO] Importing models')
model_path = 'models/model_pt9.h5'
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
model = checkpoint['model']
transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0) - 0.5)])

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

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

main_steering_angle = 0
main_speed = 5
main_brake = 0
max_speed = 30

while True:
    car_state = client.getCarState()
    if client.isApiControlEnabled() and car_state.speed < main_speed - 1:
        car_controls.throttle = 1 if car_controls.brake == 0 else 0
        car_controls.steering = main_steering_angle
        client.setCarControls(car_controls)

    if keyboard.is_pressed('q'):
        apiControl = not apiControl
        client.enableApiControl(apiControl)
        client.armDisarm(apiControl)
        if client.isApiControlEnabled():
            car_controls.brake = 0
            
        time.sleep(1)

    responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
    for response in responses:
        if response.pixels_as_float:
            print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
            airsim.write_pfm('py1.pfm', airsim.get_pfm_array(response))
        else:
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img = img1d.reshape(response.height, response.width, 3)
            
            results = yolo(img)
            res = results.xyxy[0] # [ [ [],[],[] ] ]
            
            for ele in res:
                # x1 y1 x2 y2
                if ele[0] < 130 and ele[1] < 68 and ele[2] > 120 and ele[3] > 80:
                    car_controls.brake = 1
                    car_controls.throttle = 0
                    break
                else:
                    car_controls.brake = 0

            client.setCarControls(car_controls)

            # image = img[65:-25, :, :]
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            # image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_AREA)
            # image = transformations(image)
            # image = torch.Tensor(image)
            # image = image.view(1, 3, IMG_HEIGHT, IMG_WIDTH)
            # image = Variable(image)
            # output = model(image).view(-1).data.numpy()

            # car_controls.steering = float(output[0])
            # client.setCarControls(car_controls)
            # print(f"OUTPUT = {car_controls.steering}, {main_speed}, {main_brake}")

            # labels = []
            # boxes = []
            # for ele in res:
            #     if ele[5] == 0. or ele[5] == 2. or ele[5] == 7. :
            #         if ele[5] == 0.:
            #             labels.append('Person') 
            #         elif ele[5] == 2.:
            #             labels.append('Car')
            #         elif ele[5] == 7.:
            #             labels.append('Truck')

            #         boxes.append(list(ele[:4])) 

            # print(boxes)

            # boxes = torch.tensor(boxes)

            # img = transforms.ToTensor()(img.copy())
            # img = img * 255
            # img = img.type(torch.uint8)
            # img = img.unsqueeze(0)

            # drawn_boxes = draw_bounding_boxes(image=img[0], boxes= boxes, labels=labels, width=2, colors=['blue' for _ in range(len(labels))])
            
            # tensor_to_pil = transforms.ToPILImage()(drawn_boxes.squeeze(0))
            # pic = np.array(tensor_to_pil)
            
            # cv2.imshow('detections', pic)
            # cv2.waitKey(1)

    if client.isApiControlEnabled() and (car_state.speed > main_speed):
            car_controls.throttle = 0
            client.setCarControls(car_controls)

    if (client.simGetCollisionInfo()).has_collided:
        client.reset()
        apiControl = True
        client.enableApiControl(apiControl)
        client.armDisarm(True)

