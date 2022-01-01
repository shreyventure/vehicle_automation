import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model = torch.hub.load("VCasecnikovs/Yet-Another-YOLOv4-Pytorch", "yolov4", pretrained=True)

# imgs = ['C:/Users/shreyas/OneDrive/Documents/AirSim/dataset/IMG/img_PhysXCar__0_1640694709591308200.png']  # batch of images
imgs = ['https://static.scientificamerican.com/sciam/cache/file/4059E498-B855-4281-BAC1D988ABA009A2.jpg']  # batch of images

results = model(imgs)

results.print()
results.show()  # or .save()

results.xyxy[0]  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)