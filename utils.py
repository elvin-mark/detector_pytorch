import torch
import torch.nn as nn
import torchvision
from collections import defaultdict
import xml.etree.ElementTree as ET


def collate_fn(batch):
    return tuple(zip(*batch))


def get_target(xml_path, obj_labels):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    labels_dict = {k: v for v, k in enumerate(obj_labels)}
    for child_obj in root:
        if child_obj.tag == "object":
            for child in child_obj:
                if child.tag == "name":
                    labels.append(labels_dict[child.text])
                if child.tag == "bndbox":
                    tmp = []
                    for child_ in child:
                        tmp.append(int(child_.text))
                    boxes.append(tmp)
    boxes = torch.tensor(boxes).float()
    labels = torch.tensor(labels).long()
    return {"boxes": boxes, "labels": labels}


def evaluate(model, test_dl, dev):
    model.eval()
    loss_record = defaultdict(lambda: 0)
    for x, y in test_dl:
        imgs = [x_.to(dev) for x_ in x]
        targets = [{k: v.to(dev) for k, v in y_.items()} for y_ in y]
        loss = model(imgs, targets)
        tot_loss = sum([l for l in loss.values()])
        for k, v in loss.items():
            loss_record[k] += v.item()
    for k in loss_record:
        loss_record[k] /= len(test_dl)
    return loss_record


def train_one_step(model, train_dl, optim, dev):
    model.train()
    loss_record = defaultdict(lambda: 0)
    for x, y in train_dl:
        optim.zero_grad()
        imgs = [x_.to(dev) for x_ in x]
        targets = [{k: v.to(dev) for k, v in y_.items()} for y_ in y]
        loss = model(imgs, targets)
        for k, v in loss.items():
            loss_record[k] += v.item()
        tot_loss = sum([l for l in loss.values()])
        tot_loss.backward()
        optim.step()
    for k in loss_record:
        loss_record[k] /= len(train_dl)
    return loss_record


def train(model, train_dl, test_dl, optim, epochs, dev):
    for epoch in range(epochs):
        record_ = {"epoch": epoch}
        train_loss = train_one_step(model, train_dl, optim, dev)
        for k, v in train_loss.items():
            record_["train_" + k] = v
        if test_dl is not None:
            test_loss = evaluate(model, test_dl, dev)
            for k, v in test_loss.items():
                record_["test_" + k] = v
        print(",".join([f"{k}: {v:.4f}" for k, v in record_.items()]))
