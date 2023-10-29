import tensorflow as tf
from tensorflow import keras
import keras
import numpy as np
import os
import glob
from keras.preprocessing import image
from flask import Flask, request, render_template
import PIL
import cv2
import numpy as np
from numpy.core.defchararray import join, mod
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch._C import device
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn    
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms as torchtrans  
import matplotlib.patches as patches


labels_path = 'F:\lookforgunsonpicAI\gunset\\test\Labels'
imgs_path = 'F:\lookforgunsonpicAI\gunset\\test\Images'
output_path = 'F:\lookforgunsonpicAI\git\lookforgunAI\\flask_pac3\static'
test_path = 'F:\lookforgunsonpicAI\git\lookforgunAI\\flask_pac3\images'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LOAD_MODEL_VERSION = "1.0a"
LOAD_MODEL_FILENAME = "weapon_trained_model-"+LOAD_MODEL_VERSION+".pt"
LOAD_DIR = 'F:\lookforgunsonpicAI\yolo_python\savedmodel\\'
cpu_device = torch.device("cpu")

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

test_data = gun(imgs_path, labels_path)
img,tar = test_data[11]
gunsmodel = torch.load(LOAD_DIR+LOAD_MODEL_FILENAME, map_location=torch.device('cpu'))
gunsmodel.eval()
input = []
input.append(img)
outputs = gunsmodel(input)
outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
app = Flask(__name__, static_url_path="/static")
img_size = 300
modelgun = keras.models.load_model('F:\lookforgunsonpicAI\git\lookforgunAI\\flask_pac3\lookforguns.keras')
modelhuman = keras.models.load_model('F:\lookforgunsonpicAI\git\lookforgunAI\\flask_pac3\lookforhumans.keras')

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

def process_and_save_predictions(test_path, model, device, output_path, threshold=0.5):
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


def load_and_preprocess_images(img_dir, filename):
    images = []
    path = []
    for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
        if filename in img_path:
            print(filename)
            img = image.load_img(img_path, target_size=(img_size, img_size))
            img = image.img_to_array(img)
            img = img / 255.0 
            images.append(img)
            path.append(img_path)
    images = np.array(images)
    return images, path
    
@app.route('/')
def home(): 
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    dir_dlin = 0
    file = request.files['file']
    file.save(target_dir + file.filename)
    dir_dlin = len('Загрузка файла:' + target_dir + file.filename)
    print('_' * dir_dlin)
    print('Файл заргужен:' + target_dir + file.filename)
    print('‾' * dir_dlin)
    temp, path = load_and_preprocess_images(target_dir, file.filename)

    predictionsgun = modelgun.predict(temp)
    predicted_classesgun = np.argmax(predictionsgun, axis=1)
    predictionshuman = modelhuman.predict(temp)
    predicted_classeshuman = np.argmax(predictionshuman, axis=1)

    resultgun = ""
    resulthuman = ""
    warngun = 0
    warnhuman = 0
    for i in range(len(predictionsgun)): ## тут класс в текст и уровень угрозы от 0 до 3 
        if predicted_classesgun[i] + 1 == 1:
            anw = "nogun"
            warngun = 0
        elif predicted_classesgun[i] + 1 == 2:
            anw = "shortgun"
            warngun = 1
        elif predicted_classesgun[i] + 1 == 3:
            anw = "longgun"
            warngun = 1
        else:
            anw = "???"
        resultgun = anw 
    for j in range(len(predictionshuman)):
        if warngun == 0:
            anw = "notarmed"
        elif predicted_classeshuman[j] + 1 == 1:
            anw = "nohuman"
            warnhuman = 0
        elif predicted_classeshuman[j] + 1 == 2:
            anw = "notarmed"
            warnhuman = 0
        elif predicted_classeshuman[j] + 1 == 3:
            anw = "armed"
            warnhuman = 1
        else:
            anw = "???"
        resulthuman = anw
    process_and_save_predictions(test_path, gunsmodel, device, output_path, threshold=0.5)
    image_path =  "predicted_" + file.filename
    if warnhuman > 0 or warngun > 0:
        result = "!ВНИМАНИЕ! Возможна угроза безопасности"
    else: 
        result = "Угроза не обнаружена"
    #process_and_save_predictions(test_path, model, device, output_path, threshold=0.7) - не нужна 
    return render_template('./upload.html', result=result, image_path=image_path)

if __name__ == '__main__':
    target_dir = 'F:\lookforgunsonpicAI\git\lookforgunAI\\flask_pac3\images\\'  # путь к папке куда сохраняются картинки
    app.run()
    print("off")
