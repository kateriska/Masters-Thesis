import os
import wget
import subprocess
import json
import shutil

import tensorflow as tf
import object_detection
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


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

        if (self.model == "ssd_mobilenet"):

            url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
            filename = wget.download(url)
            os.rename('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz', './pretrained_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz')
            os.chdir('./pretrained_models')
            subprocess.call(['tar', '-zxvf', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'])
            os.chdir("..")


        self.load_dataset()
        self.create_label_map()
        self.create_tf_record()
        if (self.model == "ssd_mobilenet"):
            os.mkdir('./trained_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')
            shutil.copyfile('./pretrained_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config', './trained_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config')

            config = config_util.get_configs_from_pipeline_file(os.path.join('trained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config'))
            print(config)

            pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
            with tf.io.gfile.GFile(os.path.join('trained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config'), "r") as f:
                proto_str = f.read()
                text_format.Merge(proto_str, pipeline_config)

            pipeline_config.model.ssd.num_classes = 1
            pipeline_config.train_config.batch_size = 4
            pipeline_config.train_config.fine_tune_checkpoint = os.path.join('pretrained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'checkpoint', 'ckpt-0')
            pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
            pipeline_config.train_input_reader.label_map_path= os.path.join('annotations', 'label_map.pbtxt')
            pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join('annotations', 'train.record')]
            pipeline_config.eval_input_reader[0].label_map_path = os.path.join('annotations', 'label_map.pbtxt')
            pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join('annotations', 'test.record')]

            config_text = text_format.MessageToString(pipeline_config)
            with tf.io.gfile.GFile(os.path.join('trained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config'), "wb") as f:
                f.write(config_text)

            self.train_model()

    def train_model(self):
        train_command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}".format('./models/research/object_detection/model_main_tf2.py', './trained_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8','./trained_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config', self.epochs)
        print(train_command)
        #subprocess.run(train_command, shell=True, capture_output=True)
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
        labels = [{'name':'atopic', 'id':1}]

        label_map_path = os.path.join('annotations', 'label_map.pbtxt')

        open(label_map_path, 'w').close()

        with open(label_map_path, 'a') as f:
            for label in labels:
                f.write("item {" + '\n')
                f.write("\t" + "name:'" + label.get('name') + "'" + "\n")
                f.write("\t" + "id:" + str(label.get('id')) + "\n")
                f.write("}" + "\n")
        f.close()

    def create_tf_record(self):
        subprocess.call(['python', './GenerateTFRecord/generate_tfrecord.py', '-x', os.path.join('dataset', 'train_preprocessed'), '-l', os.path.join('annotations', 'label_map.pbtxt'), '-o', os.path.join('annotations', 'train.record')])
        subprocess.call(['python', './GenerateTFRecord/generate_tfrecord.py', '-x', os.path.join('dataset', 'test_preprocessed'), '-l', os.path.join('annotations', 'label_map.pbtxt'), '-o', os.path.join('annotations', 'test.record')])
