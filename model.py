'''
Author: Katerina Fortova
Master's Thesis: Analysis of Convolutional Neural Networks for Detection and Classification of Damages in Fingerprint Images
Academic Year: 2021/22

Downloading pretrained model, model configuration, generating TF records from train and validation dataset, generating training command and evaluation on ./dataset/test_preprocessed images
Checkpoint with configured pipeline is stored in ./trained_models in subfolder based on used model

Model can be trained from downloaded checkpoint from beginning or from checkpoint trained on same task (store already trained checkpoint on same task in ./trained_models/model_name/trained_checkpoint (e.g. trained_models/efficientdet_d0_coco17_tpu-32/trained_checkpoint))
pipeline.config file with configured model parameters for training is stored in ./trained_models/model_name (e.g. trained_models/efficientdet_d0_coco17_tpu-32), changing of some parametres was used for experiments

'''

import os
import wget
import subprocess
import json
import shutil
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.cElementTree as ET
import csv
import collections
import functools
import operator

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
        if not os.path.exists(os.path.join('pretrained_models',full_model_name)):
            filename = wget.download(url)
            archive_name = full_model_name + '.tar.gz'
            os.rename(archive_name, os.path.join('pretrained_models', archive_name))
            os.chdir('./pretrained_models')
            subprocess.call(['tar', '-zxvf', archive_name])
            os.chdir("..")

    # basic configuration of model pipeline - some advanced changing of training parameters was used for experiments (see Thesis text)
    def config_pipeline(self, model, full_model_name):
        if not os.path.exists(os.path.join('trained_models',full_model_name, 'pipeline.config')):
            if os.path.isdir('./trained_models/' + full_model_name) == False:
                os.mkdir('./trained_models/' + full_model_name)
            shutil.copyfile(os.path.join('pretrained_models', full_model_name, 'pipeline.config'), os.path.join('trained_models', full_model_name, 'pipeline.config'))

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(os.path.join('trained_models', full_model_name, 'pipeline.config'), "r") as f:
            text_format.Merge(f.read(), pipeline_config)

        if self.model == "ssd_mobilenet_v2" or self.model == "ssd_resnet50" or self.model == "efficient_det_d0" or self.model == 'efficient_det_d1':
            pipeline_config.model.ssd.num_classes = 4
        elif self.model == "faster_rcnn_resnet50" or self.model == "faster_rcnn_resnet101":
            pipeline_config.model.faster_rcnn.num_classes = 4
        elif self.model == 'centernet_hourglass' or self.model == 'centernet_resnet101':
          pipeline_config.model.center_net.num_classes = 4
        pipeline_config.train_config.batch_size = 4

        # if our model of detection and classification of fingerprint damages has some checkpoint, load them and continue training, otherwise load initial checkpoint of downloaded model
        if os.path.isdir(os.path.join('trained_models', full_model_name, 'trained_checkpoint')) and tf.train.latest_checkpoint(os.path.join('trained_models', full_model_name, 'trained_checkpoint')) is not None:
            trained_model_latest_checkpoint = tf.train.latest_checkpoint(os.path.join('trained_models', full_model_name, 'trained_checkpoint'))
            pipeline_config.train_config.fine_tune_checkpoint = trained_model_latest_checkpoint
            pipeline_config.train_config.fine_tune_checkpoint_type = "full"
        else:
            pipeline_config.train_config.fine_tune_checkpoint = os.path.join('pretrained_models', full_model_name, 'checkpoint', 'ckpt-0')
            pipeline_config.train_config.fine_tune_checkpoint_type = "detection"

        pipeline_config.train_input_reader.label_map_path= os.path.join('annotations', 'label_map.pbtxt')
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join('annotations', 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = os.path.join('annotations', 'label_map.pbtxt')
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join('annotations', 'val.record')]

        pipeline_config_text = text_format.MessageToString(pipeline_config)
        with tf.io.gfile.GFile(os.path.join('trained_models', full_model_name, 'pipeline.config'), "wb") as f:
            f.write(pipeline_config_text)



    def config_model(self):
        print("TRAINING MODE")
        print("Started downloading pretrained model")
        if self.model == "ssd_mobilenet_v2":
            self.download_pretrained_model(self.model, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz')
        elif self.model == "faster_rcnn_resnet50":
            self.download_pretrained_model(self.model, 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz')
        elif self.model == "faster_rcnn_resnet101":
            self.download_pretrained_model(self.model, 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz')
        elif self.model == "ssd_resnet50":
            self.download_pretrained_model(self.model, 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz')
        elif self.model == 'efficient_det_d0':
            self.download_pretrained_model(self.model, 'efficientdet_d0_coco17_tpu-32', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz')
        elif self.model == 'efficient_det_d1':
            self.download_pretrained_model(self.model, 'efficientdet_d1_coco17_tpu-32', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz')
        elif self.model == 'centernet_hourglass':
            self.download_pretrained_model(self.model, 'centernet_hg104_512x512_coco17_tpu-8', 'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz')
        elif self.model == 'centernet_resnet101':
            self.download_pretrained_model(self.model, 'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz')

        print("Started preprocessing and splitting dataset")
        self.load_dataset()
        print("Creating label map")
        self.create_label_map()
        print("Creating TF records")
        self.create_tf_record()
        print("Processing basic pipeline configuration and generating training command")
        if self.model == "ssd_mobilenet_v2":
            self.config_pipeline(self.model, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')
            self.train_model('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')
        elif self.model == "faster_rcnn_resnet50":
            self.config_pipeline(self.model, 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8')
            self.train_model('faster_rcnn_resnet50_v1_640x640_coco17_tpu-8')
        elif self.model == "faster_rcnn_resnet101":
            self.config_pipeline(self.model, 'faster_rcnn_resnet101_v1_640x640_coco17_tpu-8')
            self.train_model('faster_rcnn_resnet101_v1_640x640_coco17_tpu-8')
        elif self.model == "ssd_resnet50":
            self.config_pipeline(self.model, 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8')
            self.train_model('ssd_resnet50_v1_fpn_640x640_coco17_tpu-8')
        elif self.model == 'efficient_det_d0':
            self.config_pipeline(self.model, 'efficientdet_d0_coco17_tpu-32')
            self.train_model('efficientdet_d0_coco17_tpu-32')
        elif self.model == 'efficient_det_d1':
            self.config_pipeline(self.model, 'efficientdet_d1_coco17_tpu-32')
            self.train_model('efficientdet_d1_coco17_tpu-32')
        elif self.model == 'centernet_hourglass':
            self.config_pipeline(self.model, 'centernet_hg104_512x512_coco17_tpu-8')
            self.train_model('centernet_hg104_512x512_coco17_tpu-8')
        elif self.model == 'centernet_resnet101':
            self.config_pipeline(self.model, 'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8')
            self.train_model('centernet_resnet101_v1_fpn_512x512_coco17_tpu-8')



    # generate train command of model
    def train_model(self, full_model_name):
        train_command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}".format(os.path.join('.','model', 'research', 'object_detection', 'model_main_tf2.py'), os.path.join('trained_models', full_model_name),os.path.join('trained_models', full_model_name, 'pipeline.config'), self.epochs)
        print("Command for training model:")
        print(train_command)

    # create label map with classes of diseases, name of classes are annotated in dataset
    def create_label_map(self):
        labels = [{'name':'atopic', 'id':1}, {'name':'verruca', 'id':2}, {'name':'dysh', 'id':3}, {'name':'psor', 'id':4}]

        label_map_path = os.path.join('annotations', 'label_map.pbtxt')

        open(label_map_path, 'w').close()

        with open(label_map_path, 'a') as f:
            for label in labels:
                f.write("item {" + '\n')
                f.write("\t" + "name:'" + label.get('name') + "'" + "\n")
                f.write("\t" + "id:" + str(label.get('id')) + "\n")
                f.write("}" + "\n")
        f.close()

    # generate TF records of test and validation dataset for training from XML annotations
    def create_tf_record(self):
        subprocess.call(['python', './GenerateTFRecord/generate_tfrecord.py', '-x', os.path.join('dataset', 'train_preprocessed'), '-l', os.path.join('annotations', 'label_map.pbtxt'), '-o', os.path.join('annotations', 'train.record')])
        subprocess.call(['python', './GenerateTFRecord/generate_tfrecord.py', '-x', os.path.join('dataset', 'val_preprocessed'), '-l', os.path.join('annotations', 'label_map.pbtxt'), '-o', os.path.join('annotations', 'val.record')])


    # test trained model on test dataset, save results into ./results folder - images with bounding box and txt file with predictions and csv file into trained_models/model_name
    def test_model(self):
        print("EVALUATION MODE - Only test trained model on test dataset")

        # Empty results folder
        for file in glob.glob('./results/*'):
            os.remove(file)

        if self.model == "ssd_mobilenet_v2":
            full_model_name = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
        elif self.model == "faster_rcnn_resnet50":
            full_model_name = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
        elif self.model == "faster_rcnn_resnet101":
            full_model_name = "faster_rcnn_resnet101_v1_640x640_coco17_tpu-8"
        elif self.model == "ssd_resnet50":
            full_model_name = "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"
        elif self.model == 'efficient_det_d0':
            full_model_name = 'efficientdet_d0_coco17_tpu-32'
        elif self.model == 'efficient_det_d1':
            full_model_name = 'efficientdet_d1_coco17_tpu-32'
        elif self.model == 'centernet_hourglass':
            full_model_name = 'centernet_hg104_512x512_coco17_tpu-8'
        elif self.model == 'centernet_resnet101':
            full_model_name = 'centernet_resnet101_v1_fpn_512x512_coco17_tpu-8'

        # configure trained model
        configs = config_util.get_configs_from_pipeline_file(os.path.join('trained_models', full_model_name, 'pipeline.config'))
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        trained_model_latest_checkpoint = tf.train.latest_checkpoint(os.path.join('trained_models', full_model_name))
        ckpt.restore(trained_model_latest_checkpoint).expect_partial()

        category_index = label_map_util.create_category_index_from_labelmap(os.path.join('annotations', 'label_map.pbtxt'))

        f = open("./results/predictions.txt", "w+") # file for predictions of tested images
        csv_path = "./experiments/" + full_model_name + ".csv"
        with open(csv_path, 'w+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([full_model_name, "Atopic (real and generated)", "Atopic (real)", "Atopic (generated)", "Verruca (real and generated)", "Verruca (real)", "Verruca (generated)", "Dysh (real and generated)", "Dysh (real)", "Dysh (generated)", "Psoriasis (real and generated)", "Psoriasis (real)", "Psoriasis (generated)", "Healthy (real and generated)", "Healthy (real)", "Healthy (generated)", "Total (real and generated)", "Total (real)", "Total (generated)"])
            csv_file.close()

        # declare names of parts of dataset
        dataset_parts = ['atopic_real', 'atopic_generated', 'verruca_real', 'verruca_generated', 'dysh_real', 'dysh_generated', 'psor_real', 'psor_generated', 'healthy_real', 'healthy_generated']

        # declare dictionary for plotting normalized IoU distributions
        iou_dict = {}
        for item in dataset_parts:
            iou_dict[item] = {}
            iou_dict[item][0.0] = 0
            iou_dict[item][0.1] = 0
            iou_dict[item][0.2] = 0
            iou_dict[item][0.3] = 0
            iou_dict[item][0.4] = 0
            iou_dict[item][0.5] = 0
            iou_dict[item][0.6] = 0
            iou_dict[item][0.7] = 0
            iou_dict[item][0.8] = 0
            iou_dict[item][0.9] = 0
            iou_dict[item][1.0] = 0

        # create dictionary for storing sums of all evaluate metrics for separate classes to compute average of these metrics in the end
        evaluate_dict = {}
        for item in dataset_parts:
            evaluate_dict[item] = {}
            evaluate_dict[item]['count'] = 0
            evaluate_dict[item]['correctly_detected_area_sum'] = 0
            evaluate_dict[item]['not_detected_annotated_area_sum'] = 0
            evaluate_dict[item]['correctly_detected_recognized_area_sum'] = 0
            evaluate_dict[item]['extra_detected_area_sum'] = 0
            evaluate_dict[item]['extra_detected_recognized_area_sum'] = 0
            evaluate_dict[item]['average_detection_score_correctly_detected_area_sum'] = 0
            evaluate_dict[item]['average_detection_score_correctly_detected_recognized_area_sum'] = 0

        set_min_score_thresh = 0.3 # minimum detection score of predicted bounding boxes - means how model is sure that bounding box belongs to particular class - bounding boxes of lower score are not used for evaluation
        # detect and clasify each image from test dataset
        #processed_counter = 0
        for file in glob.glob('./dataset/test_preprocessed/*'):
            file_substr = file.split('/')[-1]
            test_image_name = file_substr.rsplit('.', 1)[0]

            f.write("FILE NAME: " + file_substr + '\n')
            f.write("DETECTIONS: " + '\n')
            extension = os.path.splitext(file)[1][1:]

            # skip at least for now xml annotations
            if extension == 'xml':
                continue

            print(file_substr)
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

            detection_classes_tolist = detections['detection_classes'].tolist()
            detection_scores_tolist = detections['detection_scores'].tolist()
            detection_boxes_tolist = detections['detection_boxes'].tolist()

            detection_classes_tolist_filtered = []
            detection_scores_tolist_filtered = []
            detection_boxes_tolist_filtered = []
            for key, value, box in zip(detection_classes_tolist, detection_scores_tolist, detection_boxes_tolist):
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
                elif key == 3:
                    f.write('\t' + "CLASS: psoriasis" + '\n')
                    f.write('\t' + "DETECTION SCORE: " + str(round(value * 100, 2)) + " %" + '\n')
                detection_classes_tolist_filtered.append(key)
                detection_scores_tolist_filtered.append(value)
                detection_boxes_tolist_filtered.append(box)

            tree = ET.parse(os.path.join('dataset', 'test_preprocessed', test_image_name + '.xml'))
            root = tree.getroot()

            # get class of image - is it healthy fingerprint or fingerprint with some disease?
            name = "healthy"
            for class_img in root.findall('object'):
                # get class of bounding boxes if image has any, if image doesnt have any annotated bounding boxes - it is healthy image without any disease
                name = class_img.find('name').text

            # compute evaluation metrics
            correctly_detected_area, extra_detected_area = self.compute_area_test(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, root, False)
            not_detected_annotated_area = 100 - correctly_detected_area # percentage of area which is annotated but not detected
            print("Correctly detected area in % of annotated bounding boxes: " + str(correctly_detected_area))
            print("Not detected area in % of annotated bounding boxes: " + str(not_detected_annotated_area))

            # correctly detected and recognized area
            correctly_detected_recognized_area, extra_detected_recognized_area = self.compute_area_test(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, root, True)

            print("Correctly detected and recognized area in % of annotated bounding boxes: " + str(correctly_detected_recognized_area))
            print("Extra detected area in % of predicted bounding boxes: " + str(extra_detected_area))
            print("Extra detected but correctly recognized area in % of predicted bounding boxes: " + str(extra_detected_recognized_area))

            # compute average detection score of correctly detected and correctly detected and recognized pixels of bounding boxes
            average_detection_score_correctly_detected_area = self.compute_detection_scores_test(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, root, False)
            average_detection_score_correctly_detected_recognized_area = self.compute_detection_scores_test(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, root, True)

            print("Average detection score of correctly detected area in %: " + str(average_detection_score_correctly_detected_area))
            print("Average detection and recognized score of correctly detected area in %: " + str(average_detection_score_correctly_detected_area))

            print()

            # values of metrics are added to current value of these metrics in dictionary for particular class - used to compute average of metrics in the end
            #dataset_parts = ['atopic_real', 'atopic_generated', 'verruca_real', 'verruca_generated', 'dysh_real', 'dysh_generated', 'psor_real', 'psor_generated', 'healthy_real', 'healthy_generated']
            if all(x in test_image_name for x in ["atopic", "_FP_"]):
                evaluate_dict = self.add_to_evaluate_dict(evaluate_dict, 'atopic_real', correctly_detected_area, not_detected_annotated_area, correctly_detected_recognized_area, extra_detected_area, extra_detected_recognized_area, average_detection_score_correctly_detected_area, average_detection_score_correctly_detected_recognized_area)
                iou_dict = self.compute_iou(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, iou_dict, root, 'atopic_real')
            elif all(x in test_image_name for x in ["dys", "_FP_"]):
                evaluate_dict = self.add_to_evaluate_dict(evaluate_dict, 'dysh_real', correctly_detected_area, not_detected_annotated_area, correctly_detected_recognized_area, extra_detected_area, extra_detected_recognized_area, average_detection_score_correctly_detected_area, average_detection_score_correctly_detected_recognized_area)
                iou_dict = self.compute_iou(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, iou_dict, root, 'dysh_real')
            elif all(x in test_image_name for x in ["psor", "_FP_"]):
                evaluate_dict = self.add_to_evaluate_dict(evaluate_dict, 'psor_real', correctly_detected_area, not_detected_annotated_area, correctly_detected_recognized_area, extra_detected_area, extra_detected_recognized_area, average_detection_score_correctly_detected_area, average_detection_score_correctly_detected_recognized_area)
                iou_dict = self.compute_iou(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, iou_dict, root, 'psor_real')
            elif all(x in test_image_name for x in ["verruca", "_FP_"]):
                evaluate_dict = self.add_to_evaluate_dict(evaluate_dict, 'verruca_real', correctly_detected_area, not_detected_annotated_area, correctly_detected_recognized_area, extra_detected_area, extra_detected_recognized_area, average_detection_score_correctly_detected_area, average_detection_score_correctly_detected_recognized_area)
                iou_dict = self.compute_iou(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, iou_dict, root, 'verruca_real')
            elif "dys_SG" in test_image_name:
                evaluate_dict = self.add_to_evaluate_dict(evaluate_dict, 'dysh_generated', correctly_detected_area, not_detected_annotated_area, correctly_detected_recognized_area, extra_detected_area, extra_detected_recognized_area, average_detection_score_correctly_detected_area, average_detection_score_correctly_detected_recognized_area)
                iou_dict = self.compute_iou(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, iou_dict, root, 'dysh_generated')
            elif "atopic_eczema_SG" in test_image_name:
                evaluate_dict = self.add_to_evaluate_dict(evaluate_dict, 'atopic_generated', correctly_detected_area, not_detected_annotated_area, correctly_detected_recognized_area, extra_detected_area, extra_detected_recognized_area, average_detection_score_correctly_detected_area, average_detection_score_correctly_detected_recognized_area)
                iou_dict = self.compute_iou(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, iou_dict, root, 'atopic_generated')
            elif "healthy_SG" in test_image_name:
                evaluate_dict = self.add_to_evaluate_dict(evaluate_dict, 'healthy_generated', correctly_detected_area, not_detected_annotated_area, correctly_detected_recognized_area, extra_detected_area, extra_detected_recognized_area, average_detection_score_correctly_detected_area, average_detection_score_correctly_detected_recognized_area)
                iou_dict = self.compute_iou(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, iou_dict, root, 'healthy_generated')
            elif "nist_" in test_image_name:
                evaluate_dict = self.add_to_evaluate_dict(evaluate_dict, 'healthy_real', correctly_detected_area, not_detected_annotated_area, correctly_detected_recognized_area, extra_detected_area, extra_detected_recognized_area, average_detection_score_correctly_detected_area, average_detection_score_correctly_detected_recognized_area)
                iou_dict = self.compute_iou(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, iou_dict, root, 'healthy_real')
            elif "PsoriasisDamagedImg-SG" in test_image_name:
                evaluate_dict = self.add_to_evaluate_dict(evaluate_dict, 'psor_generated', correctly_detected_area, not_detected_annotated_area, correctly_detected_recognized_area, extra_detected_area, extra_detected_recognized_area, average_detection_score_correctly_detected_area, average_detection_score_correctly_detected_recognized_area)
                iou_dict = self.compute_iou(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, iou_dict, root, 'psor_generated')
            elif "SG" in test_image_name:
                evaluate_dict = self.add_to_evaluate_dict(evaluate_dict, 'verruca_generated', correctly_detected_area, not_detected_annotated_area, correctly_detected_recognized_area, extra_detected_area, extra_detected_recognized_area, average_detection_score_correctly_detected_area, average_detection_score_correctly_detected_recognized_area)
                iou_dict = self.compute_iou(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, iou_dict, root, 'verruca_generated')

            image_np_array_result = image_np_array.copy()

            # visualize result with predicted bounding boxes
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_array_result,
                detections['detection_boxes'],
                detections['detection_classes'] + 1,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=set_min_score_thresh,
                agnostic_mode=False,
                line_thickness=8)

            plt.imshow(cv2.cvtColor(image_np_array_result, cv2.COLOR_BGR2RGB))
            # image with detected bounding box is saved in results folder
            plt.savefig("./results/" + file_substr, bbox_inches='tight', pad_inches = 0)
            plt.close('all')

            '''
            processed_counter += 1
            if processed_counter > 50:
                break
            '''

        f.close()


        # print detailed average statistics for each metric, classes, real samples, generated samples
        print("========================")
        print("Average correctly detected area:")
        self.print_evaluate_metrics(evaluate_dict, 'correctly_detected_area_sum', "Average correctly detected area in %", csv_path)
        print()
        print("Average not detected annotated area:")
        self.print_evaluate_metrics(evaluate_dict, 'not_detected_annotated_area_sum', "Average not detected annotated area in %", csv_path)
        print()
        print("Average correctly detected and recognized area:")
        self.print_evaluate_metrics(evaluate_dict, 'correctly_detected_recognized_area_sum', "Average correctly detected and recognized area in %", csv_path)
        print()
        print("Average extra detected area in % of predicted bounding boxes:")
        self.print_evaluate_metrics(evaluate_dict, 'extra_detected_area_sum', "Average extra detected area in % of predicted bounding boxes", csv_path)
        print()
        print("Average extra detected but correctly recognized area in % of predicted bounding boxes:")
        self.print_evaluate_metrics(evaluate_dict, 'extra_detected_recognized_area_sum', "Average extra detected but correctly recognized area in % of predicted bounding boxes", csv_path)
        print()
        print("Average detection score of correctly detected area in %:")
        self.print_evaluate_metrics(evaluate_dict, 'average_detection_score_correctly_detected_area_sum', "Average detection score of correctly detected area in %", csv_path)
        print()
        print("Average detection score of correctly detected and recognized area in %:")
        self.print_evaluate_metrics(evaluate_dict, 'average_detection_score_correctly_detected_recognized_area_sum', "Average detection score of correctly detected and recognized area in %", csv_path)
        print()
        print("Most frequent normalized IoU value:")
        self.print_evaluate_metrics(iou_dict, 'most_frequent_iou_value', "Most frequent normalized IoU value", csv_path)

        # plot graph of normalized IoU distributions for whole test dataset
        iou_edited_dict = []
        for i in iou_dict:
            iou_edited_dict.append(iou_dict[i])
        # sum all iou values for each dataset part to get final graph of whole test dataset
        iou_all_classes = dict(functools.reduce(operator.add, map(collections.Counter, iou_edited_dict)))

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        normalized_iou_scores = iou_all_classes.keys()
        normalized_iou_scores = [str(i) for i in normalized_iou_scores]

        counts = iou_all_classes.values()
        ax.bar(normalized_iou_scores, counts)
        plt.title('Normalized IoU Distributions')
        plt.xlabel('Normalized IoU (Intersection over Union Score)')
        plt.ylabel('Count of Predicted Bounding Boxes')
        plt.savefig('./results/normalized_iou_distributions.jpg', bbox_inches='tight')
        plt.clf()
        ax.cla()



    '''
    You must evaluate IoUs directly to know how tight a model’s bounding boxes are to the underlying ground truth.
    A simple way to generate aggregate statistics about the IoU of different models is by plotting histograms. Here’s the basic recipe:
    Detect objects in a dataset using a set of models
    Compute the IoUs of every prediction
    For each prediction, store the highest IoU it has with any ground truth object
    Plot a histogram with normalized counts
    https://towardsdatascience.com/iou-a-better-detection-evaluation-metric-45a511185be1
    '''
    def compute_iou(self, test_image_name, predicted_classes, predicted_scores, predicted_boxes, iou_dict, root, dataset_part):
        for size in root.findall('size'):
            width = int (size.find('width').text)
            height = int (size.find('height').text)

        for predicted_box in predicted_boxes:
            # convert predicted coordinates to real coordinates cause predicted are normalized
            xmin_predicted = round(predicted_box[1] * width)
            ymin_predicted = round(predicted_box[0] * height)
            xmax_predicted = round(predicted_box[3] * width)
            ymax_predicted = round(predicted_box[2] * height)

            max_iou_score = 0
            ground_truth_box_with_max_iou_score = []
            for bndbox in root.findall('object/bndbox'):
                xmin = int (bndbox.find('xmin').text)
                ymin = int (bndbox.find('ymin').text)
                xmax = int (bndbox.find('xmax').text)
                ymax = int (bndbox.find('ymax').text)

                intersection_xmin = max(xmin, xmin_predicted)
                intersection_ymin = max(ymin, ymin_predicted)
                intersection_xmax = min(xmax, xmax_predicted)
                intersection_ymax = min(ymax, ymax_predicted)

                intersection_area = max(0, intersection_xmax - intersection_xmin + 1) * max(0, intersection_ymax - intersection_ymin + 1)

                ground_truth_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                predicted_area = (xmax_predicted - xmin_predicted + 1) * (ymax_predicted - ymin_predicted + 1)

                iou_score = intersection_area / float(ground_truth_area + predicted_area - intersection_area)
                #print(iou_score)

                if iou_score > max_iou_score:
                    max_iou_score = iou_score
                    ground_truth_box_with_max_iou_score.append(xmin)
                    ground_truth_box_with_max_iou_score.append(ymin)
                    ground_truth_box_with_max_iou_score.append(xmax)
                    ground_truth_box_with_max_iou_score.append(ymax)

            current_value = iou_dict[dataset_part][round(max_iou_score,1)]
            # increment value on matched normalized key value
            iou_dict[dataset_part][round(max_iou_score,1)] = current_value + 1

        return iou_dict

    def compute_area_test(self, test_image_name, predicted_classes, predicted_scores, predicted_boxes, root, correct_class_recognition):
        for size in root.findall('size'):
            width = int (size.find('width').text)
            height = int (size.find('height').text)

        if correct_class_recognition == True:
            name = "healthy"
            for class_img in root.findall('object'):
                # get class of bounding boxes if image has any, if image doesnt have any annotated bounding boxes - it is healthy image without any disease
                name = class_img.find('name').text

        all_bndboxes_pixels = np.array([[]])
        i = 0

        for bndbox in root.findall('object/bndbox'):
            xmin = int (bndbox.find('xmin').text)
            ymin = int (bndbox.find('ymin').text)
            xmax = int (bndbox.find('xmax').text)
            ymax = int (bndbox.find('ymax').text)

            # get all pixels coordinates in bounding box
            X, Y = np.mgrid[xmin:xmax, ymin:ymax]
            bndbox_pixels = np.stack(np.vstack((X.ravel(), Y.ravel())), axis=-1)

            if i == 0:
                all_bndboxes_pixels = bndbox_pixels
            else:
                all_bndboxes_pixels = np.concatenate((all_bndboxes_pixels, bndbox_pixels), axis=0)
            i += 1

        all_bndboxes_pixels = np.unique(all_bndboxes_pixels, axis=0)

        j = 0

        if predicted_boxes != []:
            for predicted_box, predicted_class in zip(predicted_boxes, predicted_classes):
                if correct_class_recognition == True:
                    # figure whether annotated bounding box is really from correct class:
                    if name == "atopic" and predicted_class != 0:
                        continue
                    elif name == "verruca" and predicted_class != 1:
                        continue
                    elif name == "dysh" and predicted_class != 2:
                        continue
                    elif name == "psor" and predicted_class != 3:
                        continue

                # convert predicted coordinates to real coordinates cause predicted are normalized
                xmin_predicted = round(predicted_box[1] * width)
                ymin_predicted = round(predicted_box[0] * height)
                xmax_predicted = round(predicted_box[3] * width)
                ymax_predicted = round(predicted_box[2] * height)

                # get all pixels coordinates in bounding box
                X_predicted, Y_predicted = np.mgrid[xmin_predicted:xmax_predicted, ymin_predicted:ymax_predicted]
                predicted_bndbox_pixels = np.stack(np.vstack((X_predicted.ravel(), Y_predicted.ravel())), axis=-1)

                if j == 0:
                    all_predicted_bndboxes_pixels = predicted_bndbox_pixels
                else:
                    all_predicted_bndboxes_pixels = np.concatenate((all_predicted_bndboxes_pixels, predicted_bndbox_pixels), axis=0)
                j += 1
        else:
            all_predicted_bndboxes_pixels = []

        if correct_class_recognition == True and j == 0:
          all_predicted_bndboxes_pixels = []
        all_predicted_bndboxes_pixels = np.unique(all_predicted_bndboxes_pixels, axis=0)

        if all_predicted_bndboxes_pixels != [] and all_bndboxes_pixels != []:
            correctly_predicted_bndboxes_pixels = self.compute_same_pixels(all_bndboxes_pixels, all_predicted_bndboxes_pixels)
            # compute extra detected pixels which are not annotated
            extra_detected_bndboxes_pixels = self.compute_extra_not_annotated_pixels(all_predicted_bndboxes_pixels, all_bndboxes_pixels)
        else:
            extra_detected_bndboxes_pixels = []

        if all_predicted_bndboxes_pixels != [] and all_bndboxes_pixels != []:
            correctly_detected_area = (correctly_predicted_bndboxes_pixels.shape[0] / all_bndboxes_pixels.shape[0]) * 100
        elif all_predicted_bndboxes_pixels.size == 0 and all_bndboxes_pixels.size == 0: # for healthy fingerprint - no annotations of diseases and no prediction
            correctly_detected_area = 100
        else: # healthy fingerprint with some prediction of disease or diseased fingerprint with no predictions at all:
            correctly_detected_area = 0

        if extra_detected_bndboxes_pixels != []: # which percent from predicted bounding boxes pixels is extra detected
            extra_detected_area = (extra_detected_bndboxes_pixels.shape[0] / all_predicted_bndboxes_pixels.shape[0]) * 100
        else: # for healthy fingerprint - no annotations of diseases and no prediction
            extra_detected_area = 0

        return correctly_detected_area, extra_detected_area

    def compute_detection_scores_test(self, test_image_name, predicted_classes, predicted_scores, predicted_boxes, root, correct_class_recognition):
        for size in root.findall('size'):
            width = int (size.find('width').text)
            height = int (size.find('height').text)

        if correct_class_recognition == True:
            name = "healthy"
            for class_img in root.findall('object'):
                # get class of bounding boxes if image has any, if image doesnt have any annotated bounding boxes - it is healthy image without any disease
                name = class_img.find('name').text

        all_bndboxes_pixels = np.array([[]])
        i = 0

        for bndbox in root.findall('object/bndbox'):
            xmin = int (bndbox.find('xmin').text)
            ymin = int (bndbox.find('ymin').text)
            xmax = int (bndbox.find('xmax').text)
            ymax = int (bndbox.find('ymax').text)

            # get all pixels coordinates in bounding box
            X, Y = np.mgrid[xmin:xmax, ymin:ymax]

            bndbox_pixels = np.stack(np.vstack((X.ravel(), Y.ravel())), axis=-1)

            if i == 0:
                all_bndboxes_pixels = bndbox_pixels
            else:
                all_bndboxes_pixels = np.concatenate((all_bndboxes_pixels, bndbox_pixels), axis=0)
            i += 1

        all_bndboxes_pixels = np.unique(all_bndboxes_pixels, axis=0)

        j = 0
        all_predicted_bndboxes_detection_scores = np.array([])

        if predicted_boxes != []:
            for predicted_box, predicted_class, predicted_detection_score in zip(predicted_boxes, predicted_classes, predicted_scores):
                if correct_class_recognition == True:
                    # figure whether annotated bounding box is really from correct class:
                    if name == "atopic" and predicted_class != 0:
                        continue
                    elif name == "verruca" and predicted_class != 1:
                        continue
                    elif name == "dysh" and predicted_class != 2:
                        continue
                    elif name == "psor" and predicted_class != 3:
                        continue

                # convert predicted coordinates to real coordinates cause predicted are normalized
                xmin_predicted = round(predicted_box[1] * width)
                ymin_predicted = round(predicted_box[0] * height)
                xmax_predicted = round(predicted_box[3] * width)
                ymax_predicted = round(predicted_box[2] * height)

                # get all pixels coordinates in bounding box
                X_predicted, Y_predicted = np.mgrid[xmin_predicted:xmax_predicted, ymin_predicted:ymax_predicted]
                predicted_bndbox_pixels = np.stack(np.vstack((X_predicted.ravel(), Y_predicted.ravel())), axis=-1)

                if j == 0:
                    all_predicted_bndboxes_pixels = predicted_bndbox_pixels
                else:
                    all_predicted_bndboxes_pixels = np.concatenate((all_predicted_bndboxes_pixels, predicted_bndbox_pixels), axis=0)

                # insert detection score of whole bounding box for every pixel in bounding box
                for i in range(0, predicted_bndbox_pixels.shape[0]):
                    all_predicted_bndboxes_detection_scores = np.append(all_predicted_bndboxes_detection_scores, predicted_detection_score)

                j += 1
        else:
            all_predicted_bndboxes_pixels = []

        if correct_class_recognition == True and j == 0:
          all_predicted_bndboxes_pixels = []
        all_predicted_bndboxes_pixels_edit = np.unique(all_predicted_bndboxes_pixels, axis=0)
        #print(all_predicted_bndboxes_pixels)
        if all_predicted_bndboxes_pixels_edit != [] and all_bndboxes_pixels != []:
            correctly_predicted_bndboxes_pixels = self.compute_same_pixels(all_bndboxes_pixels, all_predicted_bndboxes_pixels_edit)

            # stack pixels coordinates and their detection_score, each row contains pixel coordinate and detection score of pixel coordinate
            stacked_values = np.column_stack((all_predicted_bndboxes_pixels,all_predicted_bndboxes_detection_scores))

            stacked_values_pixels = stacked_values[:,[0,1]].astype(np.int64)
            filtered_indexes = self.get_indexes_of_intersecting_rows(stacked_values_pixels, correctly_predicted_bndboxes_pixels)

            if filtered_indexes != []:
                stacked_values_filtered = self.filter_largest_pixels_detection_score(stacked_values[filtered_indexes,:])
                # sum all detection scores from filtered array and compute average
                correctly_predicted_pixels_detection_scores = stacked_values_filtered[:, -1].sum()
                average_detection_score = (correctly_predicted_pixels_detection_scores / stacked_values_filtered.shape[0]) * 100
            else:
                average_detection_score = None
        else:
            average_detection_score = None;

        return average_detection_score



    def compute_same_pixels(self, A, B):
        '''
        Function for getting intersecting rows across two 2D numpy arrays
        Source:
        ***************************************************************************************
        *    Title: Get intersecting rows across two 2D numpy arrays
        *    Author: user of stackoverflow with nickname "Joe Kington" -> https://stackoverflow.com/users/325565/joe-kington
        *    Date: 29.11.2011
        *    Code version: 1
        *    Availability: https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
        **************************************************************************************
        '''

        nrows, ncols = A.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],'formats':ncols * [A.dtype]}

        C = np.intersect1d(A.view(dtype), B.view(dtype))

       # This last bit is optional if you're okay with "C" being a structured array...
        C = C.view(A.dtype).reshape(-1, ncols)
        return C


    # compute pixels which are detected but not annotated - difference of two sets
    # inspired by compute_same_pixels method
    def compute_extra_not_annotated_pixels(self, A, B):
        nrows, ncols = A.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],'formats':ncols * [A.dtype]}

        C = np.setdiff1d(A.view(dtype), B.view(dtype))

        C = C.view(A.dtype).reshape(-1, ncols)
        return C

    def filter_largest_pixels_detection_score(self, a):
        '''
        Function for getting filtered list of pixel coordinates and their detection score with filtering only biggest detection score if pixel
        is occured in input list more times

        Input array is in format [[x0,y0,d0], [x1,y1,d1], [x0,y0,d2],..]] where x,y are pixel coordinates and d is detection score
        Problem is that same pixel can have different detection score because is in several different overlapping bounding boxes
        Therefore array is filtered and if pixel is in array more times, only row with biggest detection score is filtered
        Source:
        ***************************************************************************************
        *    Title: Filter a numpy array based on largest value
        *    Author: user of stackoverflow with nickname "Jaime" -> https://stackoverflow.com/users/110026/jaime
        *    Date: 17.8.2015
        *    Code version: 1
        *    Availability: https://stackoverflow.com/questions/32052594/filter-a-numpy-array-based-on-largest-value
        **************************************************************************************
        Code is only edited because of using of numpy array of 3-dimensional vectors instead of 4-dimensional in problem of submitted question
        '''
        perm = np.lexsort(a[:, 2::-1].T)
        a_sorted = a[perm]
        last = np.concatenate((np.all(a_sorted[:-1, :2] != a_sorted[1:, :2], axis=1),
                                   [True]))
        a_unique_max = a_sorted[last]
        res = a_unique_max[np.argsort(perm[last])]

        return res

    def get_indexes_of_intersecting_rows(self, A, B):
        '''
        Function for getting indexes of intersecting rows of 2D array A with rows of 2D array B
        Indexes are calculated with respect to array A
        e.g.
        A=array([[1, 2],[3, 4],[5, 6],[7, 8],[9, 10]])
        B=array([[1, 4],[1, 2],[5, 6],[6, 3]])
        result=[0,2]
        Source:
        ***************************************************************************************
        *    Title: Get indices of intersecting rows of Numpy 2d Array
        *    Author: user of stackoverflow with nickname "Jaime" -> https://stackoverflow.com/users/110026/jaime
        *    Date: 22.5.2014
        *    Code version: 1
        *    Availability: https://stackoverflow.com/questions/23814517/get-indices-of-intersecting-rows-of-numpy-2d-array
        **************************************************************************************
        Code is only edited because in current NumPy version, dtype in view method needs to be ('int,int') instead of ('i,i') in submitted question
        '''
        # correction because of numpy view method, sometimes NumPy can create F order array and trying to create a view on an array with an incompatible memory layout
        #(https://stackoverflow.com/questions/55024259/valueerror-when-changing-to-a-larger-dtype-its-size-must-be-a-divisor-of-the-t)
        if A.flags['F_CONTIGUOUS']:  # ‘C_CONTIGUOUS’ (‘C’) - ensure a C-contiguous array
            A = np.require(A, requirements= ["C"])
        if B.flags['F_CONTIGUOUS']:  # ‘C_CONTIGUOUS’ (‘C’) - ensure a C-contiguous array
            B = np.require(B, requirements= ["C"])

        np.in1d(A.view(dtype='int,int').reshape(-1), B.view(dtype='int,int').reshape(-1))
        np.nonzero(np.in1d(A.view(dtype='int,int').reshape(-1), B.view(dtype='int,int').reshape(-1)))
        filtered_indexes = np.nonzero(np.in1d(A.view(dtype='int,int').reshape(-1), B.view(dtype='int,int').reshape(-1)))[0]
        return filtered_indexes


    # increment current values of metrics with new value
    def add_to_evaluate_dict(self, evaluate_dict, dataset_part, correctly_detected_area, not_detected_annotated_area, correctly_detected_recognized_area, extra_detected_area, extra_detected_recognized_area, average_detection_score_correctly_detected_area, average_detection_score_correctly_detected_recognized_area):
        evaluate_dict[dataset_part]['count'] = evaluate_dict[dataset_part]['count'] + 1
        evaluate_dict[dataset_part]['correctly_detected_area_sum'] = evaluate_dict[dataset_part]['correctly_detected_area_sum'] + correctly_detected_area
        evaluate_dict[dataset_part]['not_detected_annotated_area_sum'] = evaluate_dict[dataset_part]['not_detected_annotated_area_sum'] + not_detected_annotated_area
        evaluate_dict[dataset_part]['correctly_detected_recognized_area_sum'] = evaluate_dict[dataset_part]['correctly_detected_recognized_area_sum'] + correctly_detected_recognized_area
        evaluate_dict[dataset_part]['extra_detected_area_sum'] = evaluate_dict[dataset_part]['extra_detected_area_sum'] + extra_detected_area
        evaluate_dict[dataset_part]['extra_detected_recognized_area_sum'] = evaluate_dict[dataset_part]['extra_detected_recognized_area_sum'] + extra_detected_recognized_area

        if average_detection_score_correctly_detected_area != None:
            evaluate_dict[dataset_part]['average_detection_score_correctly_detected_area_sum'] = evaluate_dict[dataset_part]['average_detection_score_correctly_detected_area_sum'] + average_detection_score_correctly_detected_area
        if average_detection_score_correctly_detected_recognized_area != None:
            evaluate_dict[dataset_part]['average_detection_score_correctly_detected_recognized_area_sum'] = evaluate_dict[dataset_part]['average_detection_score_correctly_detected_recognized_area_sum'] + average_detection_score_correctly_detected_recognized_area
        return evaluate_dict

    # compute average of metrics for input classes and parts of dataset
    def compute_average_evaluate_metrics(self, evaluate_dict, dataset_parts, metrics_name):
        if metrics_name != 'most_frequent_iou_value':
            metrics_value_count = 0
            dataset_parts_count = 0
            for item in dataset_parts:
                metrics_value_count += evaluate_dict[item][metrics_name]
                dataset_parts_count += evaluate_dict[item]['count']

            if dataset_parts_count != 0:
                result = round((metrics_value_count / dataset_parts_count),3)
            else:
                # theyre no images in test folder of input part
                result = None
            return result

        else:
            metrics_value_count = 0
            iou_edited_dict = []
            for i in evaluate_dict:
                if i in dataset_parts:
                    iou_edited_dict.append(evaluate_dict[i])
            # sum all iou values for each dataset part
            iou_selected_classes = dict(functools.reduce(operator.add, map(collections.Counter, iou_edited_dict)))
            max_key = max(iou_selected_classes, key=iou_selected_classes.get)
            all_values = iou_selected_classes.values()
            max_value = max(all_values)
            result = {}
            # return most frequent value of normalized IoU and count of bounding boxes for this normalized value for dataset part(s)
            result[max_key] = max_value

            return max_key

    # print average detailed statistics of evaluation
    def print_evaluate_metrics(self, evaluate_dict, metrics_name, metrics_print_string, csv_path):
        # dataset_parts = ['atopic_real', 'atopic_generated', 'verruca_real', 'verruca_generated', 'dysh_real', 'dysh_generated', 'psor_real', 'psor_generated', 'healthy_real', 'healthy_generated']
        atopic_real_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['atopic_real', 'atopic_generated'], metrics_name)
        atopic_real = self.compute_average_evaluate_metrics(evaluate_dict, ['atopic_real'], metrics_name)
        atopic_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['atopic_generated'], metrics_name)
        verruca_real_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['verruca_real', 'verruca_generated'], metrics_name)
        verruca_real = self.compute_average_evaluate_metrics(evaluate_dict, ['verruca_real'], metrics_name)
        verruca_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['verruca_generated'], metrics_name)
        dysh_real_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['dysh_real', 'dysh_generated'], metrics_name)
        dysh_real = self.compute_average_evaluate_metrics(evaluate_dict, ['dysh_real'], metrics_name)
        dysh_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['dysh_generated'], metrics_name)
        psor_real_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['psor_real', 'psor_generated'], metrics_name)
        psor_real = self.compute_average_evaluate_metrics(evaluate_dict, ['psor_real'], metrics_name)
        psor_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['psor_generated'], metrics_name)
        healthy_real_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['healthy_real', 'healthy_generated'], metrics_name)
        healthy_real = self.compute_average_evaluate_metrics(evaluate_dict, ['healthy_real'], metrics_name)
        healthy_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['healthy_generated'], metrics_name)
        total_real_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['atopic_real', 'atopic_generated', 'verruca_real', 'verruca_generated', 'dysh_real', 'dysh_generated', 'psor_real', 'psor_generated', 'healthy_real', 'healthy_generated'], metrics_name)
        total_real = self.compute_average_evaluate_metrics(evaluate_dict, ['atopic_real', 'verruca_real', 'dysh_real', 'psor_real', 'healthy_real'], metrics_name)
        total_gen = self.compute_average_evaluate_metrics(evaluate_dict, ['atopic_generated', 'verruca_generated', 'dysh_generated', 'psor_generated', 'healthy_generated'], metrics_name)

        print("Atopic (real and generated): " + str(atopic_real_gen))
        print("Atopic (real): " + str(atopic_real))
        print("Atopic (generated): " + str(atopic_gen))
        print("Verruca (real and generated): " + str(verruca_real_gen))
        print("Verruca (real): " + str(verruca_real))
        print("Verruca (generated): " + str(verruca_gen))
        print("Dysh (real and generated): " + str(dysh_real_gen))
        print("Dysh (real): " + str(dysh_real))
        print("Dysh (generated): " + str(dysh_gen))
        print("Psoriasis (real and generated): " + str(psor_real_gen))
        print("Psoriasis (real): " + str(psor_real))
        print("Psoriasis (generated): " + str(psor_gen))
        print("Healthy (real and generated): " + str(healthy_real_gen))
        print("Healthy (real): " + str(healthy_real))
        print("Healthy (generated): " + str(healthy_gen))
        print("Total (real and generated): " + str(total_real_gen))
        print("Total (real): " + str(total_real))
        print("Total (generated): " + str(total_gen))

        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([metrics_print_string, atopic_real_gen, atopic_real, atopic_gen, verruca_real_gen, verruca_real, verruca_gen, dysh_real_gen, dysh_real, dysh_gen, psor_real_gen, psor_real, psor_gen, healthy_real_gen, healthy_real, healthy_gen, total_real_gen, total_real, total_gen])

        csv_file.close()
