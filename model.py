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

    # download pretrained model from TensorFlow Detection Model Zoo
    def download_pretrained_model(self, model, full_model_name, url):
        if not os.path.exists(os.path.join('pretrained_models','ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')):
            filename = wget.download(url)
            archive_name = full_model_name + '.tar.gz'
            os.rename(archive_name, os.path.join('pretrained_models', archive_name))
            os.chdir('./pretrained_models')
            subprocess.call(['tar', '-zxvf', archive_name])
            os.chdir("..")

    # configuration of model pipeline
    def config_pipeline(self, model, full_model_name):
        if not os.path.exists(os.path.join('trained_models',full_model_name, 'pipeline.config')):
            os.mkdir('./trained_models/' + full_model_name)
            shutil.copyfile(os.path.join('pretrained_models', full_model_name, 'pipeline.config'), os.path.join('trained_models', full_model_name, 'pipeline.config'))

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(os.path.join('trained_models', full_model_name, 'pipeline.config'), "r") as f:
            text_format.Merge(f.read(), pipeline_config)

        pipeline_config.model.ssd.num_classes = 1
        pipeline_config.train_config.batch_size = 4

        trained_model_latest_checkpoint = tf.train.latest_checkpoint(os.path.join('trained_models', full_model_name))

        # if our model of detection and classification of fingerprint damages has some checkpoint, load them and continue training, otherwise load initial checkpoint of downloaded model
        if (trained_model_latest_checkpoint is None):
            pipeline_config.train_config.fine_tune_checkpoint = os.path.join('pretrained_models', full_model_name, 'checkpoint', 'ckpt-0')
        else:
            pipeline_config.train_config.fine_tune_checkpoint = trained_model_latest_checkpoint

        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config.train_input_reader.label_map_path= os.path.join('annotations', 'label_map.pbtxt')
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join('annotations', 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = os.path.join('annotations', 'label_map.pbtxt')
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join('annotations', 'test.record')]

        pipeline_config_text = text_format.MessageToString(pipeline_config)
        with tf.io.gfile.GFile(os.path.join('trained_models', full_model_name, 'pipeline.config'), "wb") as f:
            f.write(pipeline_config_text)



    def config_model(self):

        if (self.model == "ssd_mobilenet_v2"):
            self.download_pretrained_model(self.model, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz')

        self.load_dataset()
        self.create_label_map()
        self.create_tf_record()
        if (self.model == "ssd_mobilenet_v2"):
            self.config_pipeline(self.model, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')

        self.train_model()

    # generate train command of model
    def train_model(self):
        print(os.getcwd())
        train_command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}".format(os.path.join('.','models', 'research', 'object_detection', 'model_main_tf2.py'), os.path.join('trained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'),os.path.join('trained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config'), self.epochs)
        print("Command for training model:")
        print(train_command)
        #subprocess.Popen(['python', os.path.join('models', 'research', 'object_detection', 'model_main_tf2.py'), '--model_dir=' + os.path.join('trained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'), '--pipeline_config_path=' + os.path.join('trained_models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config'), '--num_train_steps=' + str(self.epochs)])
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
        labels = [{'name':'atopic', 'id':1}, {'name':'verruca', 'id':2}, {'name':'dysh', 'id':3}]

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


    # test trained model on test dataset, save results into ./results folder - images with bounding box and txt file with predictions (at least for now)
    # TO DO: evaluate test images predicted bounding boxes and real xml annotated bounding boxes
    def test_model(self, full_model_name):
        print("Only test")
        configs = config_util.get_configs_from_pipeline_file(os.path.join('trained_models', full_model_name, 'pipeline.config'))
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        trained_model_latest_checkpoint = tf.train.latest_checkpoint(os.path.join('trained_models', full_model_name))
        ckpt.restore(trained_model_latest_checkpoint).expect_partial()

        category_index = label_map_util.create_category_index_from_labelmap(os.path.join('annotations', 'label_map.pbtxt'))

        f = open("./results/predictions.txt", "w+") # file for predictions of tested images

        # detect and clasify each image from test dataset
        for file in glob.glob('./dataset/test_preprocessed/*'):
            file_substr = file.split('/')[-1]
            f.write("FILE NAME: " + file_substr + '\n')
            f.write("DETECTIONS: " + '\n')
            extension = os.path.splitext(file)[1][1:]

            # skip at least for now xml annotations
            if extension == 'xml':
                continue

            img = cv2.imread(file)
            image_np_array = np.array(img)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np_array, 0), dtype=tf.float32)

            image, shapes = detection_model.preprocess(input_tensor)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)

            num_detections_int = int (detections.pop('num_detections'))
            detections = {key: value[0, :num_detections_int].numpy() for key, value in detections.items()}

            detections['num_detections'] = num_detections_int
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            set_min_score_thresh = 0.5

            detection_classes_tolist = detections['detection_classes'].tolist()
            detection_scores_tolist = detections['detection_scores'].tolist()


            for key, value in zip(detection_classes_tolist, detection_scores_tolist):

                # print only detections with more than e.g. 0.5 certainity
                if value <= set_min_score_thresh:
                    continue

                if key == 0:
                    f.write('\t' + "CLASS: atopic eczema" + '\n')
                    f.write('\t' + "DETECTION SCORE: " + str(round(value * 100, 2)) + " %" + '\n')

                elif key == 1:
                    f.write('\t' + "CLASS: verruca vulgaris" + '\n')
                    f.write('\t' + "DETECTION SCORE: " + str(round(value * 100, 2)) + " %" + '\n')

                elif key == 2:
                    f.write('\t' + "CLASS: dishydrosis" + '\n')
                    f.write('\t' + "DETECTION SCORE: " + str(round(value * 100, 2)) + " %" + '\n')

            image_np_array_result = image_np_array.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_array_result,
                detections['detection_boxes'],
                detections['detection_classes'] + 1,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=8,
                min_score_thresh=set_min_score_thresh,
                agnostic_mode=False,
                line_thickness=8)

            plt.imshow(cv2.cvtColor(image_np_array_result, cv2.COLOR_BGR2RGB))
            # image with detected bounding box is saved in results folder
            plt.savefig("./results/" + file_substr, bbox_inches='tight', pad_inches = 0)

        f.close()
