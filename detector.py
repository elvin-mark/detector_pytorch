import torch
import torch.nn as nn
import torchvision
from parser import create_parser
from utils import train
from datasets import create_dataloader
from optim import create_optimizer
from models import create_model

args = create_parser()

if args.gpu and torch.cuda.is_available():
    print("Using GPU")
    dev = torch.device("cuda:0")

else:
    print("Using CPU. It can be a little bit slow.")
    dev = torch.device("cpu")

print("Creating Model!")
model = create_model(args).to(dev)
optim = create_optimizer(model, args)
print("Preparing Dataset")
train_dl, test_dl, labels = create_dataloader(args)
print("Start training....")
train(model, train_dl, test_dl, optim, args.epochs, dev)

if args.save_model:
    print("Saving model...")
    torch.save(model.state_dict(), "saved_model.ckpt")
