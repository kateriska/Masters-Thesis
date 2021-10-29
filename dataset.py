import glob
import os
import cv2
import numpy as np

class Dataset:
    def __init__(self):
        super().__init__()
        self.dataset_path = "./dataset/train/*"

    def preprocess_dataset(self):
        for file in glob.glob(self.dataset_path):
            file_substr = file.split('/')[-1]
            extension = os.path.splitext(file)[1][1:]

            if extension == 'xml':
                continue

            img = cv2.imread(file, 0)
            img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

            ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((21,21), np.uint8)
            opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations

            cv2.Canny(opening, 100, 200)
            result_tresh = cv2.add(tresh_img, opening)
            result_orig = cv2.add(img, opening) # add mask with input image

            cv2.imwrite('./dataset/train_preprocessed/' + file_substr,result_orig)
