# Masters-Thesis

## Analysis of Convolutional Neural Networks for Detection and Classification of Damages in Fingerprint Images

Classification and detection of diseased fingerprints with use of modern available approaches of CNN, their tuning and comparison.

### Setup:

Install virtual environment and activate:
`python -m venv MT-env`
`source MT-env/bin/activate`

Create .gitignore folders hierarchy:
`mkdir Masters-Thesis/dataset`
`mkdir Masters-Thesis/annotations`
`mkdir Masters-Thesis/pretrained_models`
`mkdir Masters-Thesis/results`
`mkdir Masters-Thesis/trained_models`
`mkdir Masters-Thesis/model`
`mkdir Masters-Thesis/model`
`mkdir Masters-Thesis/GenerateTFRecord`
`mkdir dataset/train_preprocessed`
`mkdir dataset/test_preprocessed`

Note: Initial whole dataset is in ./dataset/train folder - images and also their .xml annotations (see Labellimg program for labeling images), test dataset is randomly separated during running

Install libraries:
`pip install opencv-python`
`pip install tensorflow`
`pip install matplotlib`
`pip install pyyaml`

Clone TensorFlow models:
`git clone https://github.com/tensorflow/models Masters-Thesis/model`

Clone GenerateTFRecord (transform from .xml annotations to tf record):
`git clone https://github.com/nicknochnack/GenerateTFRecord Masters-Thesis/GenerateTFRecord`

Install Tensorflow Object Detection API:
`apt-get install protobuf-compiler`
`cd Masters-Thesis/model/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .`

### Run program:

For training model:
`python run.py --model {choices=['ssd_mobilenet_v2', 'faster_rcnn_resnet50', 'rfcn_resnet101', 'ssd_resnet50', 'ssd_mobilenet_v1', 'faster_rcnn_nas']} --epochs {}`

For test trained model on test dataset:
`python run.py --test`
