import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
imgs = ['C:/Users/shreyas/OneDrive/Documents/AirSim/dataset/IMG/img_PhysXCar__0_1640694709591308200.png']  # batch of images

results = model(imgs)

results.print()
results.show()  # or .save()

results.xyxy[0]  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)