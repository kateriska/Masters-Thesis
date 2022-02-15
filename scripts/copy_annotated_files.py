import glob
import os
import sys
import shutil
import xml.etree.cElementTree as ET
'''
Real verruca: 42
Real dysh: 89
Real psor: 62
Real atopic: 286
'''
file_path = "/media/katerina/DATA/DB_disease_wart_dishy/real_dys_png"
train_path = "../dataset/train"
class_name = "dysh"
print(os.path.isdir(train_path))
annotated_files_count = 0
for file in glob.glob(file_path + '/*'):
    file_substr = file.split('/')[-1]
    print(file_substr)

    test_image_name = file_substr.rsplit('.', 1)[0]
    extension = os.path.splitext(file)[1][1:]

    if extension != 'png':
        continue

    try:
        print("Try parse xml")
        tree = ET.parse(file_path + "/" + test_image_name + ".xml")
        print(tree)
    except:
        continue

    root = tree.getroot()

    # get class of image - is it healthy fingerprint or fingerprint with some disease?
    name = "healthy"
    for class_img in root.findall('object'):
        # get class of bounding boxes if image has any, if image doesnt have any annotated bounding boxes - it is healthy image without any disease
        name = class_img.find('name').text
        print(name)
        if name != class_name:
            sys.stderr.write("Wrong class name in annotation for file " + test_image_name + "\n")
            exit()

    shutil.copyfile(file, train_path + "/" + file_substr)
    shutil.copyfile(file_path + "/" + test_image_name + ".xml", train_path + "/" + test_image_name + ".xml")
    annotated_files_count += 1

print("Annotated files count: " + str(annotated_files_count))
