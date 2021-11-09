import xml.etree.cElementTree as ET
import glob
import json
import os

# [{"Type": "wart", "p1": [131, 321], "p2": [179, 369]}, {"Type": "wart", "p1": [165, 278], "p2": [209, 320]}]
for file in glob.glob('./healthy_generated_dataset/*'):
    root = ET.Element("annotation")

    file_substr = file.split('/')[-1]

    extension = os.path.splitext(file)[1][1:]
    '''
    if extension != 'json':
        continue
    '''
    image_name = file_substr.rsplit('.', 1)[0]

    ET.SubElement(root, "folder").text = "healthy_png"
    ET.SubElement(root, "filename").text = image_name + ".png"
    ET.SubElement(root, "path").text = "D:\\Stazene\\healthy_png\\" + image_name + ".png"

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "416"
    ET.SubElement(size, "height").text = "560"
    ET.SubElement(size, "depth").text = "1"
    ET.SubElement(root, "segmented").text = "0"
    '''
    with open(file) as f:
        data = json.loads(f.read())

        for record in data:
            object = ET.SubElement(root, "object")
            ET.SubElement(object, "name").text = "dysh"
            ET.SubElement(object, "pose").text = "Unspecified"
            ET.SubElement(object, "truncated").text = "0"
            ET.SubElement(object, "difficult").text = "0"
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(record["p1"][0])
            ET.SubElement(bndbox, "ymin").text = str(record["p1"][1])
            ET.SubElement(bndbox, "xmax").text = str(record["p2"][0])
            ET.SubElement(bndbox, "ymax").text = str(record["p2"][1])
    '''
    tree = ET.ElementTree(root)


    tree.write("./healthy_generated_xml_files/" + image_name + ".xml")
