import os
import wget
import subprocess
import json
import shutil
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import object_detection
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


from dataset import Dataset



class UsedModel:
    def __init__(self, model, epochs, test):
        super().__init__()

        self.model = model
        self.epochs = epochs
        self.test = test


    def load_dataset(self):
        self.dataset = Dataset()
        self.dataset.preprocess_dataset()


    def config_model(self):

        if (self.model == "ssd_mobilenet"):
            url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
            filename = wget.download(url)
            os.rename('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz', os.path.join('pretrained_models','ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'))
            os.chdir('./pretrained_models')
            subprocess.call(['tar', '-zxvf', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'])
            os.chdir("..")


        self.load_dataset()
        self.create_label_map()
        self.create_tf_record()
        if (self.model == "ssd_mobilenet"):
            os.mkdir('./trained_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')
            shutil.copyfile(os.path.join('pretrained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config'), os.path.join('trained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config'))

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


    def test_model(self):
        print("Only test")
        configs = config_util.get_configs_from_pipeline_file(os.path.join('trained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config'))
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(os.path.join('trained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'ckpt-3'))).expect_partial()

        category_index = label_map_util.create_category_index_from_labelmap(os.path.join('annotations', 'label_map.pbtxt'))
        for file in glob.glob('./dataset/test_preprocessed/*'):
            file_substr = file.split('/')[-1]
            extension = os.path.splitext(file)[1][1:]

            if extension == 'xml':
                continue

            img = cv2.imread(file)
            image_np = np.array(img)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

            image, shapes = detection_model.preprocess(input_tensor)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}

            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            min_score_thresh = 0.5

            detection_classes_tolist = detections['detection_classes'].tolist()
            detection_scores_tolist = detections['detection_scores'].tolist()


            for key, value in zip(detection_classes_tolist, detection_scores_tolist):
                if value <= min_score_thresh:
                    continue

                if key == 0:
                    print("Detected atopic eczema with detection score: " + str(value * 100) + " %")

                elif key == 1:
                    print("Detected verruca with detection score: " + str(value * 100) + " %")

            label_id_offset = 1
            image_np_with_detections = image_np.copy()



            viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=8,
            min_score_thresh=0.5,
            agnostic_mode=False,
            line_thickness=8)

            plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
            plt.savefig("./results/" + file_substr, bbox_inches='tight', pad_inches = 0)
