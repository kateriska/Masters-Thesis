import glob
import os

path = "/home/katerina/Documents/Masters-Thesis/dataset/train"
for file in glob.glob(path + '/*'):
    file_substr = file.split('/')[-1]

    extension = os.path.splitext(file)[1][1:]

    if extension == 'xml':
        continue

    test_image_name = file_substr.rsplit('.', 1)[0]

    '''
    elif all(x in test_image_name for x in ["psor", "_FP_"]):
        psor_real.append(test_image_name)
    elif all(x in test_image_name for x in ["verruca", "_FP_"]):
    '''

    if all(x in test_image_name for x in ["verruca", "_FP_"]):
        print(test_image_name[8:] + ".bmp")
