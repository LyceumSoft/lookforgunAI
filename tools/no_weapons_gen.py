from PIL import Image
import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
# КОД ДЛЯ СОЗДАНИЯ БЕЛЫХ КВАДРАТОВ ДЛЯ ДАТА СЕТ
output_folderimg = "F:\lookforgunsonpicAI\gunset\gunsnew\prepair" #путь куда будут создоваться квадраты
output_folderxml = "F:\lookforgunsonpicAI\gunset\gunsnew\prepano" #путь куда будут создаваться xml файлы 
num_images = 500 # коллличество изображений
image_size = (300, 300)
start_index = 1000 # каритинки будут созранятся c этого номера

for i in range(start_index, start_index + num_images):
    img = Image.new("RGB", image_size, (255, 255, 255))
    image_filename = os.path.join(output_folderimg, f"image{i}.jpg")
    img.save(image_filename)
    doc = Document() # создание картинок с белым квадратом
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)
    folder = doc.createElement("folder")
    folder.appendChild(doc.createTextNode("images"))
    annotation.appendChild(folder) #Задаем имя и путь к файлу изображения, и сохраняем изображение с указанным именем и путем.
    filename = doc.createElement("filename")
    filename.appendChild(doc.createTextNode(f"{i}.jpg"))
    annotation.appendChild(filename)
    path = doc.createElement("path")
    path.appendChild(doc.createTextNode(os.path.abspath(image_filename)))
    annotation.appendChild(path)
    source = doc.createElement("source")
    annotation.appendChild(source)
    database = doc.createElement("database")
    database.appendChild(doc.createTextNode("Unknown"))
    source.appendChild(database)
    size = doc.createElement("size")
    annotation.appendChild(size) #Создаем xml-документ, в котором задаем необходимые элементы и их значения, такие как путь к файлу изображения, размеры, а также имя и класс объекта "nogun" (то есть "без оружия").
    width = doc.createElement("width")
    width.appendChild(doc.createTextNode(str(image_size[0])))
    size.appendChild(width)
    height = doc.createElement("height")
    height.appendChild(doc.createTextNode(str(image_size[1]))) #  Задаем имя и путь к xml файлу, и производим запись созданного xml-документа в указанный файл.
    size.appendChild(height)
    depth = doc.createElement("depth")
    depth.appendChild(doc.createTextNode("3"))
    size.appendChild(depth)
    segmented = doc.createElement("segmented")
    segmented.appendChild(doc.createTextNode("0"))
    annotation.appendChild(segmented)
    obj = doc.createElement("object")
    annotation.appendChild(obj)
    name = doc.createElement("name")
    name.appendChild(doc.createTextNode("nogun"))
    obj.appendChild(name)

    xml_filename = os.path.join(output_folderxml, f"image{i}.xml")
    print(xml_filename) 
    with open(xml_filename, "w") as f:
        f.write(doc.toprettyxml())

print(f"Создано {num_images} изображений без оружия и XML-файлов в папке {output_folderxml}")
