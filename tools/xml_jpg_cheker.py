import os
#скрипт на проверку не достающих элементов в папке с файлами xml
#cмысл данного скрипта проверить не пропустили ли вы при выделении объектов некоторые картинки
image_folder = 'C:/Users/Admin/Desktop/gun_data_picture/images/images_big' #путь к папке с изображениями
annotation_folder = 'C:/Users/Admin/Desktop/gun_data_picture/anatation' #путь к папке с xml файлами

#---КОД-ДЛЯ-ПРОВЕРКИ---

def get_files(folder):
    return os.listdir(folder)

def get_image_files():
    image_files = get_files(image_folder)
    image_files = [f.split(".")[0] for f in image_files]
    return image_files

def get_annotation_files():
    annotation_files = get_files(annotation_folder)
    annotation_files = [f.split(".")[0] for f in annotation_files]
    return annotation_files

def find_missing_file():
    image_files = get_image_files()
    annotation_files = get_annotation_files()
    missing_files = set(image_files) - set(annotation_files)
    return missing_files

if __name__ == "__main__":
    missing_files = find_missing_file()
    print("Missing file numbers: ", missing_files)