import torch
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model = torch.hub.load("VCasecnikovs/Yet-Another-YOLOv4-Pytorch", "yolov4", pretrained=True)

# imgs = ['C:/Users/shreyas/OneDrive/Documents/AirSim/dataset/IMG/img_PhysXCar__0_1640694709591308200.png']  # batch of images
imgs = ['C:/Users/shreyas/OneDrive/Desktop/nyc.jpg']

results = model(imgs)

res_panda = results.xyxy[0]

boxes = []
for ele in res_panda:
    boxes.append(list(ele[:4]))

boxes = torch.tensor(boxes)

img_pil = Image.open(r'C:/Users/shreyas/OneDrive/Desktop/nyc.jpg')
img = transforms.ToTensor()(img_pil)
img = img * 255
img = img.type(torch.uint8)
img = img.unsqueeze(0)

drawn_boxes = draw_bounding_boxes(image=img[0], boxes= boxes)

tensor_to_pil = transforms.ToPILImage()(drawn_boxes.squeeze(0))
img_pil.convert('RGBA')
img_pil.paste(tensor_to_pil, (0,0))
img_pil.show()
