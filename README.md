# Masters-Thesis

## Analysis of Convolutional Neural Networks for Detection and Classification of Damages in Fingerprint Images

Classification and detection of diseased fingerprints with use of modern available approaches of CNN, their tuning and comparison.

### Setup:

Install virtual environment and activate:  <br>
`python -m venv MT-env` <br>
`source MT-env/bin/activate` <br>

Create .gitignore folders hierarchy:
`mkdir Masters-Thesis/dataset`  <br>
`mkdir Masters-Thesis/annotations`  <br>
`mkdir Masters-Thesis/pretrained_models` <br>
`mkdir Masters-Thesis/results` <br>
`mkdir Masters-Thesis/trained_models` <br>
`mkdir Masters-Thesis/model` <br>
`mkdir Masters-Thesis/model` <br>
`mkdir Masters-Thesis/GenerateTFRecord` <br>
`mkdir dataset/train_preprocessed` <br>
`mkdir dataset/test_preprocessed` <br>

Note: Initial whole dataset is in ./dataset/train folder - images and also their .xml annotations (see Labellimg program for labeling images), test dataset is randomly separated during running

Install libraries: <br>
`pip install opencv-python`  <br>
`pip install tensorflow` <br>
`pip install matplotlib` <br>
`pip install pyyaml` <br>

Clone TensorFlow models: <br>
`git clone https://github.com/tensorflow/models Masters-Thesis/model`

Clone GenerateTFRecord (transform from .xml annotations to tf record): <br>
`git clone https://github.com/nicknochnack/GenerateTFRecord Masters-Thesis/GenerateTFRecord`

Install Tensorflow Object Detection API: <br>
`apt-get install protobuf-compiler` <br>
`cd Masters-Thesis/model/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .`

### Run program:

For training model: <br>
`python run.py --model {choices=['ssd_mobilenet_v2', 'faster_rcnn_resnet50', 'rfcn_resnet101', 'ssd_resnet50', 'ssd_mobilenet_v1', 'faster_rcnn_nas']} --epochs {}`

For test trained model on test dataset: <br>
`python run.py --test`
