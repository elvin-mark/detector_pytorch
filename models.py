import torch
import torch.nn as nn
import torchvision


def fasterrcnn(args):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=args.pretrained
    )
    model.roi_heads_box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, args.num_classes + 1
        )
    )
    return model


def create_model(args):
    if args.model == "fasterrcnn":
        return fasterrcnn(args)
    else:
        return None
