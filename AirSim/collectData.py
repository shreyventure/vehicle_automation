import airsim
import time
import cv2
import numpy as np
import keyboard

client = airsim.CarClient() # https://microsoft.github.io/AirSim/api_docs/html/_modules/airsim/client.html#CarClient
client.confirmConnection()
apiControl = False
client.enableApiControl(apiControl)
# client.armDisarm(apiControl)
car_controls = airsim.CarControls()

while True:
    # print(client.getDistanceSensorData().distance)
    # print(client.getGpsData())
    # print(client.getLidarData())

    car_state = client.getCarState()
    # print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

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

                cv2.imwrite('img.png', img_rgb)

        if (client.simGetCollisionInfo()).has_collided:
            client.reset()
            apiControl = True
            client.enableApiControl(apiControl)
            client.armDisarm(True)