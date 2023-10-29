import cv2
import numpy as np
from numpy.core.defchararray import join, mod
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
import torch
from torch._C import device
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn    
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
from torchvision import transforms as torchtrans  

labels_path = 'F:\lookforgunsonpicAI\gunset\\test\Labels'
imgs_path = 'F:\lookforgunsonpicAI\gunset\\test\Images'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class gun(Dataset):
    def __init__(self,imgs_path,labels_path):

        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.img_name = [img for img in sorted(os.listdir(self.imgs_path))]
        self.label_name = [label for label in sorted(os.listdir(self.labels_path))]

    def __getitem__(self,idx):

        image_path = os.path.join(self.imgs_path,str(self.img_name[idx]))
        img = cv2.imread(image_path)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = img_rgb/255
        img_res = torch.as_tensor(img_res).to(device)
        img_res = img_res.permute(2, 0, 1)
        
        label_name = self.img_name[idx][:-4] + "txt"
        label_path = os.path.join(self.labels_path,str(label_name))
        with open(label_path, 'r') as label_file:
            l_count = int(label_file.readline())
            box = []
            for i in range(l_count):
                box.append(list(map(int, label_file.readline().split())))

        target={}
        target["boxes"] = torch.as_tensor(box).to(device)
        area = []
        for i in range(len(box)):
           
            a = (box[i][2] - box[i][0]) * (box[i][3] - box[i][1])
            area.append(a)
        target["area"] = torch.as_tensor(area).to(device)
        labels = []
        for i in range(len(box)):
            labels.append(1)

        target["image_id"] = torch.as_tensor([idx]).to(device)
        target["labels"] = torch.as_tensor(labels, dtype = torch.int64).to(device)


        return img_res,target

    def __len__(self):
        return len(self.img_name)

def process_and_save_predictions(test_path, model, device, output_path, threshold=0.7):
    image_files = [f for f in os.listdir(test_path) if f.endswith('.jpg')]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for img_file in image_files:
        img_path = os.path.join(test_path, img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = img_rgb / 255
        img_res = torch.as_tensor(img_res).to(device).permute(2, 0, 1)
        input = [img_res]
        outputs = model(input)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        nms_prediction = apply_nms(outputs[0], iou_thresh=threshold)
        output_img_path = os.path.join(output_path, f'predicted_{img_file}')
        plot_and_save_img_bbox(torch_to_pil(img_res), nms_prediction, output_img_path)

LOAD_MODEL_VERSION = "1.0a"
LOAD_MODEL_FILENAME = "weapon_trained_model-"+LOAD_MODEL_VERSION+".pt"
LOAD_DIR = 'F:\lookforgunsonpicAI\yolo_python\savedmodel\\'

cpu_device = torch.device("cpu")
model= torch.load(LOAD_DIR+LOAD_MODEL_FILENAME, map_location=torch.device('cpu'))
model.eval()

test_path = 'F:\lookforgunsonpicAI\site\picture_save'
test_data = gun(imgs_path, labels_path)
img,tar = test_data[11]
input = []
input.append(img)
outputs = model(input)
outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

import matplotlib.patches as patches
def apply_nms(orig_prediction, iou_thresh=None):
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')

def plot_and_save_img_bbox(img, target, output_path):
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    for box in target['boxes']:
        x, y, width, height = box[0].detach().numpy(), box[1].detach().numpy(), (box[2] - box[0]).detach().numpy(), (box[3] - box[1]).detach().numpy()
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        a.add_patch(rect)
    plt.savefig(output_path)
    plt.close()

output_path = 'F:\lookforgunsonpicAI\site\\ready'
process_and_save_predictions(test_path, model, device, output_path, threshold=0.7)