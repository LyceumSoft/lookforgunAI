import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import keras
import numpy as np
import os
import tqdm
import glob
from keras.preprocessing import image
img_size = 300
model = keras.models.load_model('lookforguns.keras')

def load_and_preprocess_images(img_dir):
    images = []
    path = []
    for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = image.img_to_array(img)
        img = img / 255.0 
        images.append(img)
        path.append(img_path)
    images = np.array(images)
    return images, path
print("1 - shortgun | 2 - longgun | 3 - nogun")
filepath = "F:\lookforgunsonpicAI\gunset\gunsnew/test"
temp, path = load_and_preprocess_images(filepath)    
predictions = model.predict(temp)
print(predictions)
predicted_classes = np.argmax(predictions, axis=1)
for i in range(len(predictions)):
    print(f"{path[i]} | {predictions[i]} | класс: {predicted_classes[i]+1}")
