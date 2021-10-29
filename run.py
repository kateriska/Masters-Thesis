import argparse
from model import UsedModel

def parse_args():
    argParser = argparse.ArgumentParser(description='Train model for detection and classification of diseased fingerprints')
    argParser.add_argument('--model', dest='model', action='store', default="ssd_mobilenet", choices=['ssd_mobilenet', 'faster_rcnn'], help='Select type of model.')
    argParser.add_argument('--epochs', dest='epochs', action='store', type=int, help='Number of epochs to train for.')
    return argParser.parse_args()

if __name__ == "__main__":
    init_args = {}
    args = parse_args()
    for arg in vars(args):
        if getattr(args, arg) is not None:
            init_args[arg] = getattr(args, arg)

    print(init_args)
    model = UsedModel(**init_args)
    model.config_model()
