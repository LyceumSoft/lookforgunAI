import tensorflow as tf
from tensorflow import keras
import keras
import numpy as np
import os
import glob
from keras.preprocessing import image
from flask import Flask, request, render_template
import PIL
app = Flask(__name__)
img_size = 300
modelgun = keras.models.load_model('F:\lookforgunsonpicAI\git\lookforgunAI\\flask_pac3\lookforguns.keras')
modelhuman = keras.models.load_model('F:\lookforgunsonpicAI\git\lookforgunAI\\flask_pac3\lookforhumans.keras')
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
            warngun = 2
        else:
            anw = "???"
        resultgun = anw
    for j in range(len(predictionshuman)):
        if predicted_classeshuman[j] + 1 == 1:
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
    print(resultgun, resulthuman, (warnhuman + warngun))
    result = [resultgun, resulthuman, int(warngun + warnhuman)]
    os.remove(f"{target_dir}/{file.filename}")
    return render_template('upload.html', result=result)

if __name__ == '__main__':
    target_dir = 'F:\lookforgunsonpicAI\git\lookforgunAI\\flask_pac3\images\\'  # путь к папке куда сохраняются картинки
    app.run()
    print("off")
