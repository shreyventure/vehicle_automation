import torch.nn as nn
print('Importing tensorflow...')
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

IMG_HEIGHT = 320
IMG_WIDTH = 70

class NetworkDense(nn.Module):

    def __init__(self):
        super(NetworkDense, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        
    def forward(self, input):  
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output


class NetworkLight(nn.Module):

    def __init__(self):
        super(NetworkLight, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=48*4*19, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=3)
        )
        

    def forward(self, input):
        input = input.view(input.size(0), 3, IMG_HEIGHT, IMG_WIDTH)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

def nvidia_model():
    model = Sequential()
    model.add(layers.Lambda(lambda x: x/127.5-1.0, input_shape=(60, 100, 3)))
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
    model.add(layers.Dense(3))
    return model 