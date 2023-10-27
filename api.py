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
# Загрузка вашей модели
model = keras.models.load_model('lookforguns.keras')
def load_and_preprocess_images(img_dir):
    images = []
    for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = image.img_to_array(img)
        img = img / 255.0 
        images.append(img)
    return np.array(images)
test_image_dir = 'F:\lookforgunsonpicAI\site'
filepath = "F:\lookforgunsonpicAI\gunset\gunsnew\images"
image = load_and_preprocess_images(filepath)
print(model.predict(image))