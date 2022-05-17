'''
Author: Katerina Fortova
Master's Thesis: Analysis of Convolutional Neural Networks for Detection and Classification of Damages in Fingerprint Images
Academic Year: 2021/22

Preprocessing of dataset images from ./dataset/train folder and splitting them randomly into train, validation and test dataset into folders ./dataset/train_preprocessed, ./dataset/val_preprocessed and ./dataset/test_preprocessed
'''

import glob
import os
import cv2
import numpy as np
import shutil
import random

class Dataset:
    def __init__(self, use_used_dataset_split):
        super().__init__()
        self.dataset_path = os.path.join('dataset', 'train', '*')
        self.preprocessed_dataset_path = os.path.join('dataset', 'train_preprocessed', '*')
        self.preprocessed_dataset_val_path = os.path.join('dataset', 'val_preprocessed', '*')
        self.preprocessed_dataset_test_path = os.path.join('dataset', 'test_preprocessed', '*')
        self.use_used_dataset_split = use_used_dataset_split

    def preprocess_dataset(self):
        for file in glob.glob(os.path.abspath(self.preprocessed_dataset_path)):
            os.remove(file)

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
            kernel = np.ones((51,51), np.uint8)
            opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations

            cv2.Canny(opening, 100, 200)
            result_tresh = cv2.add(tresh_img, opening)
            result_orig = cv2.add(img, opening) # add mask with input image

            cv2.imwrite(os.path.join('dataset', 'train_preprocessed', file_substr),result_orig)

        if self.use_used_dataset_split == False:
            print("Randomly splitting dataset into train, validation and test")
            self.split_dataset()
        else:
            print("Using split of dataset which was used by author for training and experiments")
            self.use_used_dataset_split_for_experiments()


    # split dataset into test and train folder and store in train or test folder also their xml annotations
    def split_dataset(self):
        for file in glob.glob(os.path.abspath(self.preprocessed_dataset_val_path)):
            os.remove(file)

        for file in glob.glob(os.path.abspath(self.preprocessed_dataset_test_path)):
            os.remove(file)

        test_images_names = []

        atopic_real = []
        dysh_real = []
        psor_real = []
        verruca_real = []

        dysh_generated = []
        verruca_generated = []
        atopic_generated = []
        healthy_generated = []
        psor_generated = []

        nist_real = []

        for file in glob.glob(self.preprocessed_dataset_path):
            file_substr = file.split('/')[-1]
            extension = os.path.splitext(file)[1][1:]

            if extension == 'xml':
                continue

            test_image_name = file_substr.rsplit('.', 1)[0]

            if all(x in test_image_name for x in ["atopic", "_FP_"]):
                atopic_real.append(test_image_name)
            elif all(x in test_image_name for x in ["dys", "_FP_"]):
                dysh_real.append(test_image_name)
            elif all(x in test_image_name for x in ["psor", "_FP_"]):
                psor_real.append(test_image_name)
            elif all(x in test_image_name for x in ["verruca", "_FP_"]):
                verruca_real.append(test_image_name)
            elif "dys_SG" in test_image_name:
                dysh_generated.append(test_image_name)
            elif "atopic_eczema_SG" in test_image_name:
                atopic_generated.append(test_image_name)
            elif "healthy_SG" in test_image_name:
                healthy_generated.append(test_image_name)
            elif "nist_" in test_image_name:
                nist_real.append(test_image_name)
            elif "PsoriasisDamagedImg-SG" in test_image_name:
                psor_generated.append(test_image_name)
            elif "SG" in test_image_name:
                verruca_generated.append(test_image_name)



        # split real samples from each class 60 : 20 : 20 (train : val : test)
        atopic_real_train, atopic_real_val, atopic_real_test = self.split_train_val_test(atopic_real, 0.6, 0.8)
        dysh_real_train, dysh_real_val, dysh_real_test = self.split_train_val_test(dysh_real, 0.6, 0.8)
        psor_real_train, psor_real_val, psor_real_test = self.split_train_val_test(psor_real, 0.6, 0.8)
        verruca_real_train, verruca_real_val, verruca_real_test = self.split_train_val_test(verruca_real, 0.6, 0.8)

        # split generated fingerprints in ratio 80 : 10 : 10
        dysh_generated_train, dysh_generated_val, dysh_generated_test = self.split_train_val_test(dysh_generated, 0.8, 0.9)
        verruca_generated_train, verruca_generated_val, verruca_generated_test = self.split_train_val_test(verruca_generated, 0.8, 0.9)
        atopic_generated_train, atopic_generated_val, atopic_generated_test = self.split_train_val_test(atopic_generated, 0.8, 0.9)
        healthy_generated_train, healthy_generated_val, healthy_generated_test = self.split_train_val_test(healthy_generated, 0.8, 0.9)
        psor_generated_train, psor_generated_val, psor_generated_test = self.split_train_val_test(psor_generated, 0.8, 0.9)

        nist_real_train, nist_real_val, nist_real_test = self.split_train_val_test(nist_real, 0.8, 0.9)


        val_images_names = []
        test_images_names = []
        for file in glob.glob(self.preprocessed_dataset_path):
            file_substr = file.split('/')[-1]
            extension = os.path.splitext(file)[1][1:]

            if extension == 'xml':
                continue

            test_image_name = file_substr.rsplit('.', 1)[0]

            if test_image_name in atopic_real_val or test_image_name in dysh_real_val or test_image_name in psor_real_val or test_image_name in verruca_real_val or test_image_name in dysh_generated_val or test_image_name in verruca_generated_val or test_image_name in atopic_generated_val or test_image_name in healthy_generated_val or test_image_name in nist_real_val or test_image_name in psor_generated_val:
                shutil.move(file, os.path.join('dataset', 'val_preprocessed', file_substr))
                shutil.move(os.path.join('dataset', 'train_preprocessed', test_image_name + ".xml"), os.path.join('dataset', 'val_preprocessed', test_image_name + ".xml"))
                val_images_names.append(test_image_name)
            elif test_image_name in atopic_real_test or test_image_name in dysh_real_test or test_image_name in psor_real_test or test_image_name in verruca_real_test or test_image_name in dysh_generated_test or test_image_name in verruca_generated_test or test_image_name in atopic_generated_test or test_image_name in healthy_generated_test or test_image_name in nist_real_test or test_image_name in psor_generated_test:
                shutil.move(file, os.path.join('dataset', 'test_preprocessed', file_substr))
                shutil.move(os.path.join('dataset', 'train_preprocessed', test_image_name + ".xml"), os.path.join('dataset', 'test_preprocessed', test_image_name + ".xml"))

                test_images_names.append(test_image_name)
        print("Validation dataset images:")
        print(val_images_names)
        print("Test dataset images:")
        print(test_images_names)


    def split_train_val_test(self, samples, split_val_value, split_test_value):
        random.shuffle(samples)

        train, val, test = np.split(samples, [int(len(samples)* split_val_value), int(len(samples)*split_test_value)])
        return train, val, test

    # dont split dataset randomly but use same splitting into train, test and val dataset which was used by author of Thesis for training and experiments
    def use_used_dataset_split_for_experiments(self):
        val_images_names_txt = open('./dataset/val_preprocessed_split.txt', 'r')
        val_images_names = [line.strip() for line in val_images_names_txt.readlines()]

        test_images_names_txt = open('./dataset/test_preprocessed_split.txt', 'r')
        test_images_names = [line.strip() for line in test_images_names_txt.readlines()]

        for file in glob.glob(os.path.abspath(self.preprocessed_dataset_val_path)):
            os.remove(file)

        for file in glob.glob(os.path.abspath(self.preprocessed_dataset_test_path)):
            os.remove(file)

        for file in glob.glob(self.preprocessed_dataset_path):
            file_substr = file.split('/')[-1]
            extension = os.path.splitext(file)[1][1:]

            if extension == 'xml':
                continue

            test_image_name = file_substr.rsplit('.', 1)[0]

            if test_image_name in val_images_names:
                shutil.move(file, os.path.join('dataset', 'val_preprocessed', file_substr))
                shutil.move(os.path.join('dataset', 'train_preprocessed', test_image_name + ".xml"), os.path.join('dataset', 'val_preprocessed', test_image_name + ".xml"))

            elif test_image_name in test_images_names:
                shutil.move(file, os.path.join('dataset', 'test_preprocessed', file_substr))
                shutil.move(os.path.join('dataset', 'train_preprocessed', test_image_name + ".xml"), os.path.join('dataset', 'test_preprocessed', test_image_name + ".xml"))

        print("Validation dataset images:")
        print(val_images_names)
        print("Test dataset images:")
        print(test_images_names)
