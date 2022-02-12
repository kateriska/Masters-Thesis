import glob
import os

for file in glob.glob('./atopic_generated_dataset/*'):
    print(file)
    file_substr = file.split('/')[-1]
    print(file_substr)
    os.rename(file, "./atopic_generated_dataset/atopic_eczema_" + file_substr)
