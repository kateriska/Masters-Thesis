import cv2
import glob
import os
import xml.etree.cElementTree as ET

for file in glob.glob('./atopic_generated_dataset/*'):
    file_substr = file.split('/')[-1]
    print(file_substr)
    extension = os.path.splitext(file)[1][1:]

    if extension == 'json':
        continue

    image = cv2.imread(file)

    file_substr = file.split('/')[-1]
    print(file_substr)
    extension = os.path.splitext(file)[1][1:]


    test_image_name = file_substr.rsplit('.', 1)[0]
    tree = ET.parse(os.path.join('atopic_generated_xml_files', test_image_name + '.xml'))
    root = tree.getroot()

    for bndbox in root.findall('object/bndbox'):
        xmin = int (bndbox.find('xmin').text)
        ymin = int (bndbox.find('ymin').text)
        xmax = int (bndbox.find('xmax').text)
        ymax = int (bndbox.find('ymax').text)

        image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2)

    cv2.imwrite("./atopic_generated_dataset_bounding_boxes/" + file_substr, image)
