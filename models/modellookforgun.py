import tensorflow as tf
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from keras.layers import *
from keras.preprocessing import image
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
matplotlib.style.use('ggplot')

img_size = 300
batch_size = 16
EPOCHS = 35
img = "F:\lookforgunsonpicAI\gunset\gunsnew\images"
labels = "F:\lookforgunsonpicAI\gunset\gunsnew\labels"

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='Отклонения от точных данных (train)'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='Отклонения от точных данных (valid)'
    )
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='Потери при тренеровке'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='Потери при утверждении'
    )
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()

def load_and_preprocess_images(img_dir):
    images = []
    for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = image.img_to_array(img)
        img = img / 255.0 
        images.append(img)
    return np.array(images)

def load_and_preprocess_labels(labels_dir):
    labels = []
    for label_path in glob.glob(os.path.join(labels_dir, '*.xml')):
        with open(label_path, 'r') as label_file:
            xml_data = label_file.read()
            root = ET.fromstring(xml_data)
            label = root.find('object').find('name').text
            labels.append(label)
    return np.array(labels)

print("готовим данные...")
X = load_and_preprocess_images(img)
y = load_and_preprocess_labels(labels)
y = [0 if label == 'nogun' else 1 if label == 'short_gun' or label == "short_weapons" else 2 for label in y]
print("успешная загрузка")
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)
print(X_train)
print("Генерация...")

train_data_generator = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
valid_data_generator = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(batch_size)
print("Успех")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
  loss="sparse_categorical_crossentropy",
  metrics=['accuracy']
)

model.summary()

history = model.fit(train_data_generator, epochs=EPOCHS, validation_data=valid_data_generator)
model.save('F:\lookforgunsonpicAI\gunset\gunsfinal\\fast\model1\guns.keras')

train_acc = history.history['accuracy']
valid_acc = history.history['val_accuracy']
train_loss = history.history['loss']
valid_loss = history.history['val_loss']

save_plots(train_acc, valid_acc, train_loss, valid_loss)
