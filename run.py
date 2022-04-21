'''
Author: Katerina Fortova
Master's Thesis: Analysis of Convolutional Neural Networks for Detection and Classification of Damages in Fingerprint Images
Academic Year: 2021/22

Parsing arguments for training or evaluation mode

'''

import argparse
from model import UsedModel

def parse_args():
    argParser = argparse.ArgumentParser(description='Train model for detection and classification of diseased fingerprints')
    argParser.add_argument('--model', dest='model', action='store', default="ssd_mobilenet_v2", choices=['ssd_mobilenet_v2', 'faster_rcnn_resnet50', 'faster_rcnn_resnet101', 'ssd_resnet50', 'efficient_det_d0', 'efficient_det_d1', 'centernet_hourglass', 'centernet_resnet101'], help='Select type of model.')
    argParser.add_argument('--epochs', dest='epochs', default=25000, action='store', type=int, help='Number of epochs to train for.')
    argParser.add_argument('--test', dest='test', action='store_true', default=False, help='Run model on test folder, detect and clasify diseases.')
    argParser.add_argument('--use_used_dataset_split', dest='use_used_dataset_split', action='store_true', default=False, help='Use same images in train, val and test folders, as were used for Thesis experiments.')
    return argParser.parse_args()

if __name__ == "__main__":
    init_args = {}
    args = parse_args()
    for arg in vars(args):
        if getattr(args, arg) is not None:
            init_args[arg] = getattr(args, arg)

    model = UsedModel(**init_args)

    if args.test == False: # train model
        model.config_model()
    else: # load model from latest checkpoint, detect and clasify diseases on test dataset
        model.test_model()
