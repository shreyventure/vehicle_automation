import airsim
import time
import cv2
import numpy as np
import keyboard
from numpy import interp

from model import *

model_path = 'my_model.h5'
model = nvidia_model()
model.load_weights(model_path)

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
apiControl = True
client.enableApiControl(apiControl)
client.armDisarm(apiControl)
car_controls = airsim.CarControls()

main_steering_angle = 0
main_speed = 7
main_brake = 0
max_speed = 30

while True:
    # print(client.getDistanceSensorData().distance)
    # print(client.getGpsData())
    # print(client.getLidarData())

    car_state = client.getCarState()
    # print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))
    if client.isApiControlEnabled() and car_state.speed < main_speed - 1:
        # car_controls.brake = main_brake
        car_controls.throttle = 1
        car_controls.steering = main_steering_angle

    if keyboard.is_pressed('q'):
        apiControl = not apiControl
        client.enableApiControl(apiControl)
        client.armDisarm(apiControl)
        time.sleep(1)

    if client.isApiControlEnabled():
        responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
        for response in responses:
            if response.pixels_as_float:
                print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                airsim.write_pfm('py1.pfm', airsim.get_pfm_array(response))
            else:
                # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) #get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
                # img_rgb = img_rgb[65:-25, :, :]
                image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YUV)
                image = cv2.resize(image, (100, 60), cv2.INTER_AREA)
                
                
                output = model.predict(np.array([image]), batch_size=1)
                output = output[0]
                print(f'OUTPUT: {output}')

                main_steering_angle = float(output[0])
                car_controls.steering = main_steering_angle
                # main_speed = interp(float(output[1]), [0, 1], [0, max_speed])
                # main_brake = float(output[2])
                # print(f"OUTPUT = {main_steering_angle}, {main_speed}, {main_brake}")

        if client.isApiControlEnabled() and (car_state.speed > main_speed):
            car_controls.throttle = 0
            # car_controls.brake = 1

        client.setCarControls(car_controls)

        if (client.simGetCollisionInfo()).has_collided:
            client.reset()
            apiControl = True
            client.enableApiControl(apiControl)
            client.armDisarm(True)

