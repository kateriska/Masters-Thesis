import os
import wget
import subprocess
import json

from dataset import Dataset

class UsedModel:
    def __init__(self, model, epochs):
        super().__init__()

        self.model = model
        self.epochs = epochs


    def load_dataset(self):
        self.dataset = Dataset()
        self.dataset.preprocess_dataset()


    def config_model(self):
        '''
        if (self.model == "ssd_mobilenet"):
            url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
            filename = wget.download(url)
            os.rename('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz', './pretrained_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz')
            os.chdir('./pretrained_models')
            subprocess.call(['tar', '-zxvf', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'])
        '''
        self.load_dataset()
        self.create_label_map()

    '''
    label map in format:
    item {
	name:'eczema'
	id:1
    }
    item {
	   name:'verruca'
	   id:2
    }
    '''


    def create_label_map(self):
        labels = [{'name':'eczema', 'id':1}, {'name':'verruca', 'id':2}]

        label_map_path = './annotations/label_map.pbtxt'

        open(label_map_path, 'w').close()

        with open(label_map_path, 'a') as f:
            for label in labels:
                f.write("item {" + '\n')
                f.write("\t" + "name:'" + label.get('name') + "'" + "\n")
                f.write("\t" + "id:" + str(label.get('id')) + "\n")
                f.write("}" + "\n")
        f.close()
