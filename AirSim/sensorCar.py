import airsim
import time
import cv2
import numpy as np
import keyboard
from numpy import interp

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from model import *

print('[INFO] Importing model')
model_path = 'models/model_pt.h5'
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
model = checkpoint['model']
transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0) - 0.5)])

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

print('[INFO] Connecting to the server')
client = airsim.CarClient() # https://microsoft.github.io/AirSim/api_docs/html/_modules/airsim/client.html#CarClient
client.confirmConnection()
apiControl = True
client.enableApiControl(apiControl)
client.armDisarm(apiControl)
car_controls = airsim.CarControls()

main_steering_angle = 0
main_speed = 8
main_brake = 0
max_speed = 30

while True:
    available_distance = client.getDistanceSensorData().distance
    print(available_distance, car_controls.brake)
    # print(client.getGpsData())
    # print(client.getLidarData())

    car_state = client.getCarState()
    if client.isApiControlEnabled() and car_state.speed < main_speed - 1:
        car_controls.throttle = 1
        car_controls.steering = main_steering_angle
        client.setCarControls(car_controls)

    if keyboard.is_pressed('q'):
        apiControl = not apiControl
        client.enableApiControl(apiControl)
        client.armDisarm(apiControl)
        time.sleep(1)

    if client.isApiControlEnabled():
        if available_distance < 15:
            car_controls.throttle = 0
            car_controls.brake = 1
            client.setCarControls(car_controls)
        else:
            car_controls.brake = 0
        # responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
        # for response in responses:
        #     if response.pixels_as_float:
        #         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        #         airsim.write_pfm('py1.pfm', airsim.get_pfm_array(response))
        #     else:
        #         img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) #get numpy array
        #         img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
        #         image = img_rgb[65:-25, :, :]
        #         # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        #         image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_AREA)
        #         image = transformations(image)
        #         image = torch.Tensor(image)
        #         image = image.view(1, 3, IMG_HEIGHT, IMG_WIDTH)
        #         image = Variable(image)
        #         output = model(image).view(-1).data.numpy()

        #         car_controls.steering = float(output[0])
        #         client.setCarControls(car_controls)
        #         print(f"OUTPUT = {car_controls.steering}, {main_speed}, {main_brake}")

        if client.isApiControlEnabled() and (car_state.speed > main_speed):
            car_controls.throttle = 0
            client.setCarControls(car_controls)

        if (client.simGetCollisionInfo()).has_collided:
            client.reset()
            apiControl = True
            client.enableApiControl(apiControl)
            client.armDisarm(True)

