import glob
import os

for file in glob.glob('./nist_dataset/*'):
    print(file)
    file_substr = file.split('/')[-1]
    print(file_substr)
    os.rename(file, "./nist_dataset/nist_" + file_substr)
