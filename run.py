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
    argParser.add_argument('--num_train_steps', dest='num_train_steps', default=25000, action='store', type=int, help='Number of training steps to train for.')
    argParser.add_argument('--test', dest='test', action='store_true', default=False, help='Run model on test folder, detect and clasify diseases.')
    argParser.add_argument('--use_used_dataset_split', dest='use_used_dataset_split', action='store_true', default=False, help='Use same images in train, val and test folders, as were used for Thesis experiments.')
    '''
    set --use_ckpt when you want to start training from your own trained checkpoint on same task or when you want to set path for trained model used for evaluation
    TRAINING MODE - default ckpt path when --use_ckpt is set is Masters-Thesis/trained_models/model_name/trained_checkpoint (e.g. Masters-Thesis/trained_models/efficientdet_d0_coco17_tpu-32/trained_checkpoint)
                  - when --use_ckpt is not set, ckpt is loaded from downloaded pretrained model
                  - you can specify your own path with --ckpt_path <PATH> (if its different from Masters-Thesis/trained_models/model_name/trained_checkpoint)
    EVALUATION MODE - when --ckpt_path is not specified, path is Masters-Thesis/trained_models/model_name (e.g. Masters-Thesis/trained_models/efficientdet_d0_coco17_tpu-32)
                    - when --ckpt_path is specified, path is set to specified path
    '''
    argParser.add_argument('--use_ckpt', dest='use_ckpt', action='store_true', default=False, help='When set, checkpoint model trained on same task is used for training.')
    argParser.add_argument('--ckpt_path', dest='ckpt_path', action='store', default="", help="Path to users given checkpoint.")

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
