import glob
import os

path = "/media/katerina/DATA/dataset_splits/val_preprocessed_split/content/Masters-Thesis/dataset/val_preprocessed"
for file in glob.glob(path + '/*'):
    file_substr = file.split('/')[-1]

    extension = os.path.splitext(file)[1][1:]

    if extension == 'xml':
        continue
    test_image_name = file_substr.rsplit('.', 1)[0]
    print(test_image_name)
