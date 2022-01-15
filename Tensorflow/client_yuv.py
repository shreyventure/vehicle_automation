import socketio # __version__ : 4.2.1
import eventlet # __version__ : 0.31.0
from flask import Flask # __version__ : 1.1.2
import base64
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

sio = socketio.Server()
 
app = Flask(__name__) # '__main__'
speed_limit = 10
model = None #tf.keras.models.load_model('my_model_yuv4.h5')
model_path = 'my_model_yuv4.h5'

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED

def nvidia_model():
    model = Sequential()
    model.add(layers.Lambda(lambda x: x/127.5-1.0, input_shape=(100, 60, 3)))
    # We have a series of 3 5x5 convolutional layers with a stride of 2x2
    model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))    
    model.add(layers.BatchNormalization())

    
    # This is then followed by 2 3x3 convolutional layers with a 1x1 stride
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')) 
    model.add(layers.BatchNormalization())

    # model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')) 
    # model.add(layers.BatchNormalization())
    
    # Flattening the output of last convolutional layer before entering fully connected phase
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(1164, activation='relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.BatchNormalization())
    
    # Output layer
    model.add(layers.Dense(2))
    return model 

def generate_model():
    global model

    model = nvidia_model()

    # model.add(layers.Lambda(lambda x: x/127.5-1.0, input_shape=(66, 200, 3)))

    # model.add(layers.Conv2D(24, (5, 5), strides = (2, 2), activation='elu'))
    # model.add(layers.Conv2D(36, (5, 5), strides = (2, 2), activation='elu'))
    # model.add(layers.Conv2D(48, (5, 5), strides = (2, 2), activation='elu'))
    # model.add(layers.Conv2D(64, (3, 3), activation='elu'))
    # model.add(layers.Conv2D(64, (3, 3), activation='elu'))

    # model.add(layers.Dropout(rate=0.3))

    # model.add(layers.Flatten())

    # model.add(layers.Dense(100, activation='elu'))

    # model.add(layers.Dense(50, activation='elu'))

    # model.add(layers.Dense(10, activation='elu'))

    # model.add(layers.Dense(2))

    model.load_weights(model_path)

def convert_img(img):
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = img[60:-25, :, :]
  img = cv2.resize(img, (60, 100), cv2.INTER_AREA)
  return img

# connection
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    emit_control(0, 0, 0)

# input
@sio.on('telemetry')
def telemetry(sid, data):
    try:
        if data:
            steering_angle = float(data["steering_angle"])
            throttle = float(data["throttle"])
            speed = float(data["speed"])
            image = data["image"]
            img_b64decode = base64.b64decode(image)
            img_array = np.frombuffer(img_b64decode, np.uint8)
            img=cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)
            img = convert_img(img)
            # cv2.imshow("view",img)

            # cv2.waitKey(5)
            data = model.predict(np.array([img]), batch_size=1)

            new_steering_angle = float(data[0][0])
            if -0.05 < new_steering_angle < 0.05:
                new_steering_angle = 0.0
            # new_speed = float(data[0][1])
            if abs(new_steering_angle - steering_angle) > 0.6:
                brake = 1
            else:
                brake = 0

            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - new_steering_angle**2 - (speed/speed_limit)**2 - 0.3

            print(f'{new_steering_angle}\t{throttle}')
            del img

            emit_control(new_steering_angle, throttle, brake)
        else:
            sio.emit('manual', data={}, skip_sid=True)
    except Exception as e:
        print("Error: "+ e)
        # cv2.destroyAllWindows()

# output
def emit_control(steering_angle, throttle, brake):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__(),
        'brake': brake.__str__()
    })
 
if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    generate_model()
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)