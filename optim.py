import torch


def create_optimizer(model, args):
    if args.optim == "SGD":
        return torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        return None
