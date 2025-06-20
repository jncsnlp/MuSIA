#!/usr/bin/env python
import argparse
import torch
from list_dataset import ImageFilelist
import numpy as np
import pickle
from tqdm import tqdm

from os.path import dirname
import os
import torchvision as tv
import timm

# from torchvision import datasets as dset


def parse_args():
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("data_root", help="Path to data")
    parser.add_argument("out_file", help="Path to output file")
    parser.add_argument("model", help="Path to config")
    parser.add_argument(
        "--checkpoint",
        # default="checkpoints/resnet50d.pth",
        help="Path to checkpoint",
    )
    parser.add_argument("--img_list", default=None, help="Path to image list")
    parser.add_argument("--batch", type=int, default=256, help="Path to data")  # 256
    parser.add_argument("--workers", type=int, default=4, help="Path to data")
    parser.add_argument("--fc_save_path", default=None, help="Path to save fc")

    return parser.parse_args()


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    model = timm.create_model(args.model, pretrained=True, num_classes=0).cuda().eval()
    # num_classes=10是一个分类网络，num_classes=0是特征提取网络，没有最后的分类层
    print(model)
    # print(model.fc.weight.shape, model.fc.bias.shape)

    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize((224, 224)),
            # tv.transforms.Resize((32, 32)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if args.img_list is not None:
        dataset = ImageFilelist(args.data_root, args.img_list, transform)
    else:
        dataset = tv.datasets.ImageFolder(args.data_root, transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    features = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x.cuda()  # resnet50d

            feat_batch = model(x).cpu().numpy()
            # print(x.shape, feat_batch.shape)
            features.append(feat_batch)

    features = np.concatenate(features, axis=0)

    print("saving feature")
    # mmcv.mkdir_or_exist(dirname(args.out_file))
    os.makedirs(dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "wb") as f:
        pickle.dump(features, f)
    print("finished")

    if args.fc_save_path is not None:
        # 保存线性层的w和b
        print("saving w and b")
        model = timm.create_model(args.model, pretrained=True)
        # mmcv.mkdir_or_exist(dirname(args.fc_save_path))
        os.makedirs(dirname(args.fc_save_path), exist_ok=True)
        if args.model in ["swin_base_patch4_window7_224", "repvgg_b3"]:
            w = model.head.fc.weight.cpu().detach().numpy()
            b = model.head.fc.bias.cpu().detach().numpy()
        elif args.model in ["deit_base_patch16_224"]:
            w = model.head.weight.cpu().detach().numpy()
            b = model.head.bias.cpu().detach().numpy()
        else:
            w = model.fc.weight.cpu().detach().numpy()
            b = model.fc.bias.cpu().detach().numpy()
            print(w.shape, b.shape)
        with open(args.fc_save_path, "wb") as f:
            pickle.dump([w, b], f)
        print("finished")


if __name__ == "__main__":
    main()
