import cv2
import numpy as np

file = "/home/katerina/Documents/Masters-Thesis/dataset/train/atopic_eczema_FP_00007.png"
img = cv2.imread(file, 0)
img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
cv2.imwrite("norm.png", img)

ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("thresh.png", tresh_img)
# noise removal
kernel = np.ones((51,51), np.uint8)
opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations
cv2.imwrite("opening.png", opening)
cv2.Canny(opening, 100, 200)
result_tresh = cv2.add(tresh_img, opening)
result_orig = cv2.add(img, opening) # add mask with input image
cv2.imwrite("result.png", result_orig)

#cv2.imwrite(os.path.join('dataset', 'train_preprocessed', file_substr),result_orig)
