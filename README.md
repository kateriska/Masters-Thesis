# Masters-Thesis

## Analysis of Convolutional Neural Networks for Detection and Classification of Damages in Fingerprint Images

Classification and detection of diseased fingerprints with use of modern available approaches of CNN, their tuning and comparison.

**Author:** Katerina Fortova <br />
**Master's Thesis:** Analysis of Convolutional Neural Networks for Detection and Classification of Damages in Fingerprint Images <br />
**Academic Year:** 2021/22 <br />
**Google Colab Project Demo (the easiest way how to run submitted WIS archive):** https://colab.research.google.com/drive/1YgnBWWaUrqjOrwPRzjGdAFcfYkiwkypY?usp=sharing <br />
**Github repository link:** https://github.com/kateriska/Masters-Thesis

### Setup:

**Install virtual environment and activate:**  <br />
`python -m venv MT-env` <br />
`source MT-env/bin/activate` <br />

**Create .gitignore folders hierarchy:**  <br />
`mkdir Masters-Thesis/dataset`  <br />
`mkdir Masters-Thesis/annotations`  <br />
`mkdir Masters-Thesis/pretrained_models` <br />
`mkdir Masters-Thesis/results` <br />
`mkdir Masters-Thesis/trained_models` <br />
`mkdir Masters-Thesis/model` <br />
`mkdir Masters-Thesis/GenerateTFRecord` <br />
`mkdir Masters-Thesis/dataset/train_preprocessed` <br />
`mkdir Masters-Thesis/dataset/val_preprocessed` <br />
`mkdir Masters-Thesis/dataset/test_preprocessed` <br />

**Important:** Private STRaDe dataset with real diseased fingerprints can't be part of submitted Thesis. Please contact STRaDe research group for providing the dataset.

**Processing STRaDe dataset:**  <br />
Store provided STRaDe dataset images in `Masters-Thesis/strade_dataset` in appropriate subfolder based on disease. Each disease subfolder contains .txt file with images which were part of Thesis dataset.  <br />
`python process_strade_dataset.py`  <br />
This script filters used images from STRaDe dataset, renames them (for simplify of getting name of disease of image for evaluation) and converts images from bmp to png. <br />
Processed STRaDe dataset images are converted into folder `Masters-Thesis/dataset/train`, next to their already stored annotations.

**Note:** Initial whole dataset is in `Masters-Thesis/dataset/train` folder - images and also their .xml annotations (see Labellimg program for labeling images). During configuration for training, whole dataset is preprocessed and randomly splitted into train, validation and test dataset (folders `Masters-Thesis/dataset/train_preprocessed`, `Masters-Thesis/dataset/val_preprocessed` and `Masters-Thesis/dataset/test_preprocessed`).

**Install libraries:** <br />
`pip install opencv-python`  <br />
`pip install tensorflow` <br />
`pip install matplotlib` <br />
`pip install pyyaml` <br />

**Clone TensorFlow models:** <br />
`git clone https://github.com/tensorflow/models Masters-Thesis/model`

**Apply small patch to cloned TensorFlow models:** <br />
`git apply model_repo.patch`

**Clone GenerateTFRecord (transform from .xml annotations to TF record):** <br />
`git clone https://github.com/nicknochnack/GenerateTFRecord Masters-Thesis/GenerateTFRecord`

**Install Tensorflow Object Detection API:** <br />
`apt-get install protobuf-compiler` <br />
`cd Masters-Thesis/model/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .`

### Run program:

**TRAINING MODE - For configuration and training selected model:** <br />
Firstly configure all necessary for training:  <br />
`python run.py --model {choices=['ssd_mobilenet_v2', 'faster_rcnn_resnet50', 'faster_rcnn_resnet101', 'ssd_resnet50', 'efficient_det_d0', 'efficient_det_d1', 'centernet_hourglass', 'centernet_resnet101']} --num_train_steps {} --use_used_dataset_split --use_ckpt --ckpt_path <PATH>` <br />
When `--use_used_dataset_split` is set, train, validation and test dataset contain the same images as were used for training the model and experiments. Otherwise, when `--use_used_dataset_split` isn't set, the dataset is splitted randomly.<br />
Default ckpt path when `--use_ckpt` is set is `Masters-Thesis/trained_models/model_name/trained_checkpoint` (e.g. `Masters-Thesis/trained_models/efficientdet_d0_coco17_tpu-32/trained_checkpoint`)<br />
When `--use_ckpt` is not set, ckpt is loaded from downloaded pretrained model.<br />
You can specify your own path with `--ckpt_path <PATH>` (if its different from `Masters-Thesis/trained_models/model_name/trained_checkpoint`)<br />
Then training can be runned for chosen model:  <br />
`python model/research/object_detection/model_main_tf2.py --model_dir=trained_models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 --pipeline_config_path=trained_models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config`

**EVALUATION MODE - For test trained model on test dataset** (folder `Masters-Thesis/dataset/test_preprocessed`)**:** <br />
`python run.py --test --model {choices=['ssd_mobilenet_v2', 'faster_rcnn_resnet50', 'faster_rcnn_resnet101', 'ssd_resnet50', 'efficient_det_d0', 'efficient_det_d1', 'centernet_hourglass', 'centernet_resnet101'] --use_ckpt --ckpt_path <PATH>`

When `--ckpt_path` is not specified, path is `Masters-Thesis/trained_models/model_name` (e.g. `Masters-Thesis/trained_models/efficientdet_d0_coco17_tpu-32`)<br />
When `--ckpt_path` is specified, path is set to specified `<PATH>`.<br />
CSV file with results of evaluation, test dataset images with predicted bounding boxes, normalized IoU graph are stored in `Masters-Thesis/results`.
