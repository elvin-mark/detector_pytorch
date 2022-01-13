import torch
import torchvision
import os
import PIL
from utils import collate_fn, get_target


class DetectorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        transforms=torchvision.transforms.ToTensor(),
        train=True,
        get_target=None,
    ):
        self.root = args.root
        self.transforms = transforms
        self.ds_type = "train" if train else "test"
        tmp_path = os.path.join(self.root, self.ds_type)
        if os.path.exists(tmp_path):
            assert f"{self.ds_type} folder does not exit"
        self.imgs = list(sorted(os.listdir(tmp_path)))
        tmp_path = os.path.join(self.root, "annotations")
        if os.path.exists(tmp_path):
            assert "annotations does not exit"
        self.annotations = list(sorted(os.listdir(tmp_path)))
        self.get_target = get_target

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.ds_type, self.imgs[idx])
        annotation_path = os.path.join(
            self.root, "annotations", self.annotations[idx])
        img = PIL.Image.open(open(img_path, "rb")).convert("RGB")
        target = self.get_target(annotation_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


def create_dataloader(args):
    with open(os.path.join(args.root, "labels.txt")) as f:
        labels = f.readlines()
    labels = [label_.replace("\n", "") for label_ in labels]
    def get_target_(xml_path): return get_target(xml_path, labels)
    train_ds = DetectorDataset(args, get_target=get_target_)
    test_ds = DetectorDataset(args, train=False, get_target=get_target)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, collate_fn=collate_fn
    )
    if len(test_ds):
        test_dl = torch.utils.data.DataLoader(test_ds)
    else:
        test_dl = None
        print("Training without testing")
    return train_dl, test_dl, labels
