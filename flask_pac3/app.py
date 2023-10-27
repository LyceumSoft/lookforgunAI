from flask import request, render_template
from flask import Flask
from flask import render_template
import os

#осталось нейронку подсоеденить с нейронкой
target_dir = 'C:/Users/Admin/Desktop/FLASK/flask_pac3/picture_save' # поменяй путь к папке куда сохраняем картинки



target_dir = target_dir + '/'
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    dir_dlin = 0
    file = request.files['file']
    file.save(target_dir + file.filename)

    #Output - console
    dir_dlin = len('file_download:' + target_dir + file.filename)
    print('_' * dir_dlin)
    print('file_download:' + target_dir + file.filename)
    print('‾' *dir_dlin)
    #Output - console

    return render_template('upload.html')


if __name__ == '__main__':
    app.run()

