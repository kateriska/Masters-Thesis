import glob
import os

path = "/media/katerina/DATA/Stazene/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_5_filtered"
for file in glob.glob(path + '/*'):
    print(file)
    file_substr = file.split('/')[-1]
    print(file_substr)
    os.rename(file, path + "/nist_" + file_substr)
