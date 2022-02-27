import cv2
import glob
import os

file_path = "./sfinge_partial_original"
output_path = "./sfinge_partial_resized"

for file in glob.glob(file_path + '/*'):
    file_substr = file.split('/')[-1]
    print(file_substr)

    test_image_name = file_substr.rsplit('.', 1)[0]
    extension = os.path.splitext(file)[1][1:]

    image = cv2.imread(file)

    resized_down = cv2.resize(image, (320,440), interpolation= cv2.INTER_AREA)
    cv2.imwrite(output_path + "/" + file_substr, resized_down )
