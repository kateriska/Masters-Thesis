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

    # configuration of model pipeline
    def config_pipeline(self, model, full_model_name):
        if not os.path.exists(os.path.join('trained_models',full_model_name, 'pipeline.config')):
            os.mkdir('./trained_models/' + full_model_name)
            shutil.copyfile(os.path.join('pretrained_models', full_model_name, 'pipeline.config'), os.path.join('trained_models', full_model_name, 'pipeline.config'))

        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(os.path.join('trained_models', full_model_name, 'pipeline.config'), "r") as f:
            text_format.Merge(f.read(), pipeline_config)

        if (self.model == "ssd_mobilenet_v2"):
            pipeline_config.model.ssd.num_classes = 4
        elif self.model == "faster_rcnn_resnet50":
            pipeline_config.model.faster_rcnn.num_classes = 4
        pipeline_config.train_config.batch_size = 4

        if os.path.isdir(os.path.join('trained_models', full_model_name, 'trained_checkpoint')) and tf.train.latest_checkpoint(os.path.join('trained_models', full_model_name, 'trained_checkpoint')) is not None:
            trained_model_latest_checkpoint = tf.train.latest_checkpoint(os.path.join('trained_models', full_model_name, 'trained_checkpoint'))
            pipeline_config.train_config.fine_tune_checkpoint = trained_model_latest_checkpoint
            pipeline_config.train_config.fine_tune_checkpoint_type = "full"
        else:
            pipeline_config.train_config.fine_tune_checkpoint = os.path.join('pretrained_models', full_model_name, 'checkpoint', 'ckpt-0')
            pipeline_config.train_config.fine_tune_checkpoint_type = "detection"

        # if our model of detection and classification of fingerprint damages has some checkpoint, load them and continue training, otherwise load initial checkpoint of downloaded model
        '''
        if (trained_model_latest_checkpoint is None):
            pipeline_config.train_config.fine_tune_checkpoint = os.path.join('pretrained_models', full_model_name, 'checkpoint', 'ckpt-0')
        else:
            pipeline_config.train_config.fine_tune_checkpoint = trained_model_latest_checkpoint
        '''
        #pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config.train_input_reader.label_map_path= os.path.join('annotations', 'label_map.pbtxt')
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join('annotations', 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = os.path.join('annotations', 'label_map.pbtxt')
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join('annotations', 'val.record')]

        pipeline_config_text = text_format.MessageToString(pipeline_config)
        with tf.io.gfile.GFile(os.path.join('trained_models', full_model_name, 'pipeline.config'), "wb") as f:
            f.write(pipeline_config_text)



    def config_model(self):

        if self.model == "ssd_mobilenet_v2":
            self.download_pretrained_model(self.model, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz')
        elif self.model == "faster_rcnn_resnet50":
            self.download_pretrained_model(self.model, 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz')
        self.load_dataset()
        self.create_label_map()
        self.create_tf_record()
        if self.model == "ssd_mobilenet_v2":
            self.config_pipeline(self.model, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')
            self.train_model('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')
        elif self.model == "faster_rcnn_resnet50":
            self.config_pipeline(self.model, 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8')
            self.train_model('faster_rcnn_resnet50_v1_640x640_coco17_tpu-8')



    # generate train command of model
    def train_model(self, full_model_name):
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

    def create_tf_record(self):
        subprocess.call(['python', './GenerateTFRecord/generate_tfrecord.py', '-x', os.path.join('dataset', 'train_preprocessed'), '-l', os.path.join('annotations', 'label_map.pbtxt'), '-o', os.path.join('annotations', 'train.record')])
        subprocess.call(['python', './GenerateTFRecord/generate_tfrecord.py', '-x', os.path.join('dataset', 'val_preprocessed'), '-l', os.path.join('annotations', 'label_map.pbtxt'), '-o', os.path.join('annotations', 'val.record')])


    # test trained model on test dataset, save results into ./results folder - images with bounding box and txt file with predictions (at least for now)
    # TO DO: evaluate test images predicted bounding boxes and real xml annotated bounding boxes
    def test_model(self):
        print("Only test trained model on test dataset")

        # Empty results folder
        for file in glob.glob('./results/*'):
            os.remove(file)

        if self.model == "ssd_mobilenet_v2":
            full_model_name = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
        elif (self.model == "faster_rcnn_resnet50"):
            full_model_name = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'

        # configure trained model
        configs = config_util.get_configs_from_pipeline_file(os.path.join('trained_models', full_model_name, 'pipeline.config'))
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        trained_model_latest_checkpoint = tf.train.latest_checkpoint(os.path.join('trained_models', full_model_name))
        ckpt.restore(trained_model_latest_checkpoint).expect_partial()

        category_index = label_map_util.create_category_index_from_labelmap(os.path.join('annotations', 'label_map.pbtxt'))

        f = open("./results/predictions.txt", "w+") # file for predictions of tested images

        # declare dictionary for plotting normalized IoU distributions
        iou_stats = {0.0 : 0, 0.1 : 0, 0.2 : 0, 0.3 : 0, 0.4 : 0, 0.5 : 0, 0.6 : 0, 0.7 : 0, 0.8 : 0, 0.9 : 0, 1.0 : 0}

        # count of images from each class for separate statistics
        healthy_count = 0
        atopic_count = 0
        dysh_count = 0
        verruca_count = 0
        psor_count = 0

        # count of correctly detected area percentage for each class for separate statistics
        healthy_correctly_detected_area_sum = 0
        atopic_correctly_detected_area_sum = 0
        dysh_correctly_detected_area_sum = 0
        verruca_correctly_detected_area_sum = 0
        psor_correctly_detected_area_sum = 0

        healthy_not_detected_annotated_area_sum = 0
        atopic_not_detected_annotated_area_sum = 0
        dysh_not_detected_annotated_area_sum = 0
        verruca_not_detected_annotated_area_sum = 0
        psor_not_detected_annotated_area_sum = 0

        healthy_correctly_detected_recognized_area_sum = 0
        atopic_correctly_detected_recognized_area_sum = 0
        dysh_correctly_detected_recognized_area_sum = 0
        verruca_correctly_detected_recognized_area_sum = 0
        psor_correctly_detected_recognized_area_sum = 0

        healthy_extra_detected_area_sum = 0
        atopic_extra_detected_area_sum = 0
        dysh_extra_detected_area_sum = 0
        verruca_extra_detected_area_sum = 0
        psor_extra_detected_area_sum = 0
        # detect and clasify each image from test dataset
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

            set_min_score_thresh = 0.4

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

            iou_stats = self.compute_iou(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, iou_stats, root)
            #print(iou_stats)

            # plot graph of normalized IoU distributions for whole test dataset - updated after each processed image
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            normalized_iou_scores = iou_stats.keys()
            normalized_iou_scores = [str(i) for i in normalized_iou_scores]
            #print(normalized_iou_scores)
            counts = iou_stats.values()
            ax.bar(normalized_iou_scores, counts)
            plt.title('Normalized IoU Distributions')
            plt.xlabel('Normalized IoU (Intersection over Union Score)')
            plt.ylabel('Count of Predicted Bounding Boxes')
            plt.show()
            plt.savefig('./results/normalized_iou_distributions.jpg', bbox_inches='tight')
            plt.clf()
            ax.cla()

            correctly_detected_area, extra_detected_area = self.compute_area_test(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, root, False)
            not_detected_annotated_area = 100 - correctly_detected_area # percentage of area which is annotated but not detected
            print("Correctly detected area in % of annotated bounding boxes: " + str(correctly_detected_area))
            print("Not detected area in % of annotated bounding boxes: " + str(not_detected_annotated_area))

            # correctly detected and recognized area
            correctly_detected_recognized_area, extra_detected_recognized_area = self.compute_area_test(test_image_name, detection_classes_tolist_filtered, detection_scores_tolist_filtered, detection_boxes_tolist_filtered, root, True)

            print("Correctly detected and recognized area in % of annotated bounding boxes: " + str(correctly_detected_recognized_area))
            print("Extra detected area in % of whole image: " + str(extra_detected_area))
            print()

            if name == "atopic":
                atopic_count += 1
                atopic_correctly_detected_area_sum += correctly_detected_area
                atopic_not_detected_annotated_area_sum += not_detected_annotated_area
                atopic_correctly_detected_recognized_area_sum += correctly_detected_recognized_area
                atopic_extra_detected_area_sum += extra_detected_area
            elif name == "dysh":
                dysh_count += 1
                dysh_correctly_detected_area_sum += correctly_detected_area
                dysh_not_detected_annotated_area_sum += not_detected_annotated_area
                dysh_correctly_detected_recognized_area_sum += correctly_detected_recognized_area
                dysh_extra_detected_area_sum += extra_detected_area
            elif name == "verruca":
                verruca_count += 1
                verruca_correctly_detected_area_sum += correctly_detected_area
                verruca_not_detected_annotated_area_sum += not_detected_annotated_area
                verruca_correctly_detected_recognized_area_sum += correctly_detected_recognized_area
                verruca_extra_detected_area_sum += extra_detected_area
            elif name == "psor":
                psor_count += 1
                psor_correctly_detected_area_sum += correctly_detected_area
                psor_not_detected_annotated_area_sum += not_detected_annotated_area
                psor_correctly_detected_recognized_area_sum += correctly_detected_recognized_area
                psor_extra_detected_area_sum += extra_detected_area
            else:
                healthy_count += 1
                healthy_correctly_detected_area_sum += correctly_detected_area
                healthy_not_detected_annotated_area_sum += not_detected_annotated_area
                healthy_correctly_detected_recognized_area_sum += correctly_detected_recognized_area
                healthy_extra_detected_area_sum += extra_detected_area

            image_np_array_result = image_np_array.copy()

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

        f.close()

        print("Average correctly detected area for individual classes:")
        print("Atopic: " + str(atopic_correctly_detected_area_sum / atopic_count))
        print("Dysh: " + str(dysh_correctly_detected_area_sum / dysh_count))
        print("Verruca: " + str(verruca_correctly_detected_area_sum / verruca_count))
        print("Healthy: " + str(healthy_correctly_detected_area_sum / healthy_count))
        print("Psoriasis: " + str(psor_correctly_detected_area_sum / psor_count))
        print()
        print("Average not detected annotated area for individual classes:")
        print("Atopic: " + str(atopic_not_detected_annotated_area_sum/ atopic_count))
        print("Dysh: " + str(dysh_not_detected_annotated_area_sum / dysh_count))
        print("Verruca: " + str(verruca_not_detected_annotated_area_sum / verruca_count))
        print("Healthy: " + str(healthy_not_detected_annotated_area_sum / healthy_count))
        print("Psoriasis: " + str(psor_not_detected_annotated_area_sum / psor_count))
        print()
        print("Average correctly detected and recognized area for individual classes:")
        print("Atopic: " + str(atopic_correctly_detected_recognized_area_sum / atopic_count))
        print("Dysh: " + str(dysh_correctly_detected_recognized_area_sum / dysh_count))
        print("Verruca: " + str(verruca_correctly_detected_recognized_area_sum / verruca_count))
        print("Healthy: " + str(healthy_correctly_detected_recognized_area_sum / healthy_count))
        print("Psoriasis: " + str(psor_correctly_detected_recognized_area_sum / psor_count))
        print()
        print("Average extra detected area for individual classes in % of image:")
        print("Atopic: " + str(atopic_extra_detected_area_sum / atopic_count))
        print("Dysh: " + str(dysh_extra_detected_area_sum / dysh_count))
        print("Verruca: " + str(verruca_extra_detected_area_sum / verruca_count))
        print("Healthy: " + str(healthy_extra_detected_area_sum / healthy_count))
        print("Psoriasis: " + str(psor_extra_detected_area_sum / psor_count))


    '''
    You must evaluate IoUs directly to know how tight a model’s bounding boxes are to the underlying ground truth.
    A simple way to generate aggregate statistics about the IoU of different models is by plotting histograms. Here’s the basic recipe:
    Detect objects in a dataset using a set of models
    Compute the IoUs of every prediction
    For each prediction, store the highest IoU it has with any ground truth object
    Plot a histogram with normalized counts
    https://towardsdatascience.com/iou-a-better-detection-evaluation-metric-45a511185be1
    '''
    def compute_iou(self, test_image_name, predicted_classes, predicted_scores, predicted_boxes, iou_stats, root):
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


            current_value = iou_stats[round(max_iou_score,1)]
            iou_stats[round(max_iou_score,1)] = current_value + 1
        return iou_stats

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

            X, Y = np.mgrid[xmin:xmax, ymin:ymax]

            bndbox_pixels = np.stack(np.vstack((X.ravel(), Y.ravel())), axis=-1)

            if i == 0:
                all_bndboxes_pixels = bndbox_pixels
            else:
                all_bndboxes_pixels = np.concatenate((all_bndboxes_pixels, bndbox_pixels), axis=0)
            i += 1

        all_bndboxes_pixels = np.unique(all_bndboxes_pixels, axis=0)
        #print(all_bndboxes_pixels)
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
        #print(all_predicted_bndboxes_pixels)
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

        if extra_detected_bndboxes_pixels != []:
            extra_detected_area = (extra_detected_bndboxes_pixels.shape[0] / (width * height)) * 100
        else: # for healthy fingerprint - no annotations of diseases and no prediction
            extra_detected_area = 0

        return correctly_detected_area, extra_detected_area



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
