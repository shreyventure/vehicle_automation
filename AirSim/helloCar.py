import airsim
import time
import cv2
import numpy as np
import keyboard

client = airsim.CarClient() # https://microsoft.github.io/AirSim/api_docs/html/_modules/airsim/client.html#CarClient
client.confirmConnection()
apiControl = True
client.enableApiControl(apiControl)
client.armDisarm(apiControl)
car_controls = airsim.CarControls()
desired_speed = 12

while True:
    car_state = client.getCarState()
    print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

    if car_state.speed < desired_speed - 1:
        car_controls.brake = 0
        car_controls.steering = 0
        car_controls.throttle = 1
        client.setCarControls(car_controls)

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
    responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
    for response in responses:
        if response.pixels_as_float:
            print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
            airsim.write_pfm('py1.pfm', airsim.get_pfm_array(response))
        else:
            print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) #get numpy array
            img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
            image = img_rgb[65:-25, :, :]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            image = cv2.resize(image, (200, 66), cv2.INTER_AREA)
            cv2.imwrite('numpy.png', image)

    if (car_state.speed > desired_speed):
        car_controls.throttle = 0
        car_controls.brake = 1
        client.setCarControls(car_controls)

    if (client.simGetCollisionInfo()).has_collided:
        client.reset()
        apiControl = True
        client.enableApiControl(apiControl)
        client.armDisarm(True)

    if keyboard.is_pressed('q'):
        apiControl = not apiControl
        client.enableApiControl(apiControl)
        client.armDisarm(apiControl)

