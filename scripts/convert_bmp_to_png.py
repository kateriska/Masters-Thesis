from PIL import Image
import glob

for file in glob.glob('/media/katerina/DATA/DB_disease_psor/nemoci psor/*'):
    img = Image.open(file)
    #print(file)
    file_substr = file.split('/')[-1]
    #print(file_substr)
    image_name = file_substr.rsplit('.', 1)[0]
    new_file_path = "/media/katerina/DATA/DB_disease_psor/real_psor_png/psor_" + image_name + ".png"
    print(new_file_path)
    img.save(new_file_path, 'png')
