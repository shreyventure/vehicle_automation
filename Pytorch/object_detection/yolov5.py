from cProfile import label
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
from PIL import Image
import cv2
from colors import colors
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model = torch.hub.load("VCasecnikovs/Yet-Another-YOLOv4-Pytorch", "yolov4", pretrained=True)

img_path = 'C:/Users/shreyas/OneDrive/Desktop/nyc.jpg'
img = cv2.imread(img_path)

results = model(img)

res = results.xyxy[0]
labels = results.pandas().xyxy[0]['name']

boxes = []
for ele in res:
    boxes.append(list(ele[:4]))

boxes = torch.tensor(boxes)

with Image.open(img_path) as img_pil:
    img = transforms.ToTensor()(img_pil)
    img = img * 255
    img = img.type(torch.uint8)
    img = img.unsqueeze(0)

    drawn_boxes = draw_bounding_boxes(image=img[0], boxes= boxes, labels=results.pandas().xyxy[0]['name'], colors=colors.getColors(labels), width=2)
    tensor_to_pil = transforms.ToPILImage()(drawn_boxes.squeeze(0))

    pic = np.array(tensor_to_pil)
    scale_percent = 60 # percent of original size
    width = int(pic.shape[1] * scale_percent / 100)
    height = int(pic.shape[0] * scale_percent / 100)
    dim = (width, height)
    pic = cv2.resize(pic, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow('detections', pic)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()
