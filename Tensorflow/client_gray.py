import socketio # __version__ : 4.2.1
import eventlet # __version__ : 0.31.0
from flask import Flask # __version__ : 1.1.2
import base64
import cv2
import numpy as np
import tensorflow as tf

sio = socketio.Server()
 
app = Flask(__name__) # '__main__'
speed_limit = 10
model = tf.keras.models.load_model('my_model_gray9.h5')

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED

def convert_img(img):
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img = img[60:-25, :]
  img = cv2.resize(img, (200, 66), cv2.INTER_AREA)
  return img

# connection
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    emit_control(0, 0)

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
            data = model.predict(np.array([img]), batch_size=1)

            new_steering_angle = float(data[0][0])
            # new_speed = float(data[0][1])
            if -0.05 < new_steering_angle < 0.05:
                new_steering_angle = 0.0

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

# output
def emit_control(steering_angle, throttle, brake=0):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__(),
        'brake': brake.__str__()
    })
 
if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)