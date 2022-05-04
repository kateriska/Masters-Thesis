import glob
import os

TRAIN_PATH = "/media/katerina/DATA/Stazene/new_train_dataset" # folder of whole dataset used for Thesis


for file in glob.glob(TRAIN_PATH + '/*'):
    file_substr = file.split('/')[-1]
    test_image_name = file_substr.rsplit('.', 1)[0]
    extension = os.path.splitext(file)[1][1:]

    # skip txt file with used dataset imgs
    if extension == 'xml':
        continue


    if all(x in test_image_name for x in ["atopic", "_FP_"]):
        print("STRADE a")
        print(file_substr)
        os.remove(file)
    elif all(x in test_image_name for x in ["dys", "_FP_"]):
        print("STRADE d")
        os.remove(file)
    elif all(x in test_image_name for x in ["psor", "_FP_"]):
        print("STRADE p")
        os.remove(file)
    elif all(x in test_image_name for x in ["verruca", "_FP_"]):
        print("STRADE v")
        os.remove(file)
