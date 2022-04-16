'''
Author: Katerina Fortova
Master's Thesis: Analysis of Convolutional Neural Networks for Detection and Classification of Damages in Fingerprint Images
Academic Year: 2021/22

Script for processing private STRaDe dataset, which can't be part of submitted Thesis
Please contact research group STRaDe (https://strade.fit.vutbr.cz/en/) for providing dataset

Prerequisities:
STRaDe dataset images are stored in ./strade_dataset in appropriate subfolder based on disease

Each disease subfolder contains .txt file with images which were part of Thesis dataset
This script filters used images from STRaDe dataset, renames them (for simplify of getting name of disease of img for evaluation) and converts images from bmp to png
Processed STRaDe dataset images are converted into folder ./dataset/train (specified in TRAIN_PATH), next to their already stored annotations
'''

from PIL import Image
import glob
import os

STRADE_DATASET_PATH = "/media/katerina/DATA/strade_dataset"
STRADE_ATOPIC_PATH = "/media/katerina/DATA/strade_dataset/atopic"
STRADE_VERRUCA_PATH = "/media/katerina/DATA/strade_dataset/verruca"
STRADE_DYSH_PATH = "/media/katerina/DATA/strade_dataset/dysh"
STRADE_PSOR_PATH = "/media/katerina/DATA/strade_dataset/psor"
TRAIN_PATH = "/media/katerina/DATA/strade_dataset/train" # folder of whole dataset used for Thesis

# process of atopic eczema images from STRaDe dataset
atopic_txt = open(STRADE_ATOPIC_PATH + "/atopic_strade.txt", 'r')
atopic_used_imgs = [line.strip() for line in atopic_txt.readlines()]

print("Start processing atopic STRaDe dataset")
for file in glob.glob(STRADE_ATOPIC_PATH + '/*'):
    file_substr = file.split('/')[-1]
    extension = os.path.splitext(file)[1][1:]

    # skip txt file with used dataset imgs
    if extension == 'txt':
        continue

    if file_substr in atopic_used_imgs:
        img = Image.open(file)
        image_name = file_substr.rsplit('.', 1)[0]
        new_file_path = TRAIN_PATH + "/atopic_eczema_" + image_name + ".png"
        #print(new_file_path)
        img.save(new_file_path, 'png')

# process of verruca images from STRaDe dataset
verruca_txt = open(STRADE_VERRUCA_PATH + "/verruca_strade.txt", 'r')
verruca_used_imgs = [line.strip() for line in verruca_txt.readlines()]

print("Start processing verruca STRaDe dataset")
for file in glob.glob(STRADE_VERRUCA_PATH + '/*'):
    file_substr = file.split('/')[-1]
    extension = os.path.splitext(file)[1][1:]

    # skip txt file with used dataset imgs
    if extension == 'txt':
        continue

    if file_substr in verruca_used_imgs:
        img = Image.open(file)
        image_name = file_substr.rsplit('.', 1)[0]
        new_file_path = TRAIN_PATH + "/verruca_" + image_name + ".png"
        #print(new_file_path)
        img.save(new_file_path, 'png')

# process of dyshidrosis images from STRaDe dataset
dysh_txt = open(STRADE_DYSH_PATH + "/dysh_strade.txt", 'r')
dysh_used_imgs = [line.strip() for line in dysh_txt.readlines()]

print("Start processing dyshidrosis STRaDe dataset")
for file in glob.glob(STRADE_DYSH_PATH + '/*'):
    file_substr = file.split('/')[-1]
    extension = os.path.splitext(file)[1][1:]

    # skip txt file with used dataset imgs
    if extension == 'txt':
        continue

    if file_substr in dysh_used_imgs:
        img = Image.open(file)
        image_name = file_substr.rsplit('.', 1)[0]
        new_file_path = TRAIN_PATH + "/dys_" + image_name + ".png"
        #print(new_file_path)
        img.save(new_file_path, 'png')

# process of psoriasis images from STRaDe dataset
psor_txt = open(STRADE_PSOR_PATH + "/psor_strade.txt", 'r')
psor_used_imgs = [line.strip() for line in psor_txt.readlines()]

print("Start processing psoriasis STRaDe dataset")
for file in glob.glob(STRADE_PSOR_PATH + '/*'):
    file_substr = file.split('/')[-1]
    extension = os.path.splitext(file)[1][1:]

    # skip txt file with used dataset imgs
    if extension == 'txt':
        continue

    if file_substr in psor_used_imgs:
        img = Image.open(file)
        image_name = file_substr.rsplit('.', 1)[0]
        new_file_path = TRAIN_PATH + "/psor_" + image_name + ".png"
        #print(new_file_path)
        img.save(new_file_path, 'png')

print("Finished, STRaDe dataset converted into " + TRAIN_PATH)
