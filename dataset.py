import glob
import os
import cv2
import numpy as np
import shutil
import random

class Dataset:
    def __init__(self):
        super().__init__()
        self.dataset_path = os.path.join('dataset', 'train', '*')
        self.preprocessed_dataset_path = os.path.join('dataset', 'train_preprocessed', '*')
        self.preprocessed_dataset_test_path = os.path.join('dataset', 'test_preprocessed', '*')

    def preprocess_dataset(self):
        for file in glob.glob(self.dataset_path):
            file_substr = file.split('/')[-1]
            extension = os.path.splitext(file)[1][1:]

            if extension == 'xml':
                shutil.copyfile(file, os.path.join('dataset', 'train_preprocessed', file_substr))
                continue

            img = cv2.imread(file, 0)
            img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

            ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((23,23), np.uint8)
            opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations

            cv2.Canny(opening, 100, 200)
            result_tresh = cv2.add(tresh_img, opening)
            result_orig = cv2.add(img, opening) # add mask with input image

            cv2.imwrite(os.path.join('dataset', 'train_preprocessed', file_substr),result_orig)

        self.split_dataset()

    def split_dataset(self):
        #if len(os.listdir(os.path.abspath('./dataset/train_preprocessed/')) ) == 0:
        #    print("Directory is empty")
        #else:
        print(os.path.abspath(self.preprocessed_dataset_test_path))
        for file in glob.glob(os.path.abspath(self.preprocessed_dataset_test_path)):
            os.remove(file)

        test_images_names = []
        for file in glob.glob(self.preprocessed_dataset_path):
            file_substr = file.split('/')[-1]
            extension = os.path.splitext(file)[1][1:]

            if extension == 'xml':
                continue

            generated_random_number = random.random()

            if (generated_random_number >= 0.8):
                shutil.move(file, os.path.join('dataset', 'test_preprocessed', file_substr))
                test_image_name = file_substr.rsplit('.', 1)[0]
                test_images_names.append(test_image_name)

        print(test_images_names)

        for file in glob.glob(self.preprocessed_dataset_path):
            file_substr = file.split('/')[-1]
            test_xml_name = file_substr.rsplit('.', 1)[0]

            if test_xml_name in test_images_names:
                shutil.move(file, os.path.join('dataset', 'test_preprocessed', file_substr))
