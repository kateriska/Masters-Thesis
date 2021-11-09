import glob
import os

for file in glob.glob('./healthy_generated_dataset/*'):
    print(file)
    file_substr = file.split('/')[-1]
    print(file_substr)
    os.rename(file, "./healthy_generated_dataset/healthy_" + file_substr)
