import tensorflow as tf
from tensorflow import keras
import keras
import numpy as np
import os
import glob
from keras.preprocessing import image
from flask import Flask, request, render_template

app = Flask(__name)
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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    dir_dlin = 0
    file = request.files['file']
    file.save(target_dir + file.filename)

    # Output - console
    dir_dlin = len('file_download:' + target_dir + file.filename)
    print('_' * dir_dlin)
    print('file_download:' + target_dir + file.filename)
    print('‾' * dir_dlin)
    # Output - console

    temp, path = load_and_preprocess_images(target_dir)
    predictions = model.predict(temp)
    predicted_classes = np.argmax(predictions, axis=1)
    result = []

    for i in range(len(predictions)):
        if predicted_classes[i] + 1 == 1:
            anw = "nogun"
        elif predicted_classes[i] + 1 == 2:
            anw = "shortgun"
        elif predicted_classes[i] + 1 == 3:
            anw = "longgun"
        else:
            anw = "???"
        result.append((path[i], predictions[i], predicted_classes[i] + 1, anw))

    return render_template('upload.html', result=result)

if __name__ == '__main__':
    target_dir = '..git/lookforgunAI/flask_pac3/images'  # путь к папке куда сохраняются картинки
    app.run()
