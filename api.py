import os
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
model = keras.models.load_model('kerasguns.keras')
data_dir = 'F:/lookforgunsonpicAI/gunset/guns/'
test_image_dir = 'F:\lookforgunsonpicAI\site\picture_save'
with open(os.path.join(data_dir, 'label.txt')) as label_file:
    classes = label_file.read().strip().split('\n')
test_image_paths = [os.path.join(test_image_dir, filename) for filename in os.listdir(test_image_dir)]
for image_path in test_image_paths:
    img = keras.preprocessing.image.load_img(image_path, target_size=(300, 300))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    print(predicted_class)
    class_name = classes[predicted_class]
    print(f"Изображение {image_path}: Предсказанный класс - {class_name}")