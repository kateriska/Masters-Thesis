import glob
import os
import sys
import shutil
import xml.etree.cElementTree as ET
import numpy as np
'''
Real verruca: 42
Real dysh: 89
Real psor: 62
Real atopic: 286
'''

def get_folder_images_names(path):
    folder_images_names = []
    for file in glob.glob(path + '/*'):
        file_substr = file.split('/')[-1]


        test_image_name = file_substr.rsplit('.', 1)[0]
        extension = os.path.splitext(file)[1][1:]

        if extension != 'png':
            continue
        folder_images_names.append(test_image_name)
    return folder_images_names

file_path = "./atopic_real_new_annotations"
train_path = "../dataset/train_preprocessed"
val_path = "../dataset/val_preprocessed"
test_path = "../dataset/test_preprocessed"
class_name = "atopic"

annotated_files_count = 0

all_images_names = []
train_images_names = get_folder_images_names(train_path)
val_images_names = get_folder_images_names(val_path)
test_images_names = get_folder_images_names(test_path)

# check whether image is only in one folder - train, test or val
all_images_names.append(train_images_names)
all_images_names.append(val_images_names)
all_images_names.append(test_images_names)

all_images_names = list(np.concatenate(all_images_names).flat)
is_in_more_folders = (len(all_images_names) != len(set(all_images_names)))
if is_in_more_folders == True:
    sys.stderr.write("Some image is in more than one folder (train, test or val)\n")
    exit()



for file in glob.glob(file_path + '/*'):
    file_substr = file.split('/')[-1]
    print(file_substr)

    test_image_name = file_substr.rsplit('.', 1)[0]
    extension = os.path.splitext(file)[1][1:]

    '''
    if extension != 'png':
        continue
    '''
    try:
        print("Try parse xml")
        #tree = ET.parse(file_path + "/" + test_image_name + ".xml")
        tree = ET.parse(file)
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

    if test_image_name in train_images_names:
        shutil.copyfile(file, train_path + "/" + file_substr)
    elif test_image_name in val_images_names:
        shutil.copyfile(file, val_path + "/" + file_substr)
    elif test_image_name in test_images_names:
        shutil.copyfile(file, test_path + "/" + file_substr)
    #shutil.copyfile(file, train_path + "/" + file_substr)
    #shutil.copyfile(file_path + "/" + test_image_name + ".xml", train_path + "/" + test_image_name + ".xml")
    annotated_files_count += 1

print("Annotated files count: " + str(annotated_files_count))
