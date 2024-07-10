import os
import argparse
import torch
from tqdm import tqdm
from vit.data import HAM10000
from torch.utils.data import DataLoader
from model import ViT
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)

parser.add_argument("--kernel_size", default=32, type=int)
parser.add_argument("--h_dim", default=768, type=int)

parser.add_argument("--num_workers", default=32, type=int)

IMAGE_DIR = "./dataset/imgs"
LABEL_PATH = "./dataset/label.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def test(model, data_loader):
    model.eval()
    accr = 0

    with tqdm(data_loader, desc="Test") as pbar:
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            accr += (pred.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()

    accr /= len(data_loader.dataset)
    return accr


def main(args):
    dataset = HAM10000(
        image_dir=IMAGE_DIR,
        label_path=LABEL_PATH,
        kernel_size=args.kernel_size,
        train=False,
    )

    test_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    model = ViT(seq_len=13 * 13, in_features=32 * 32 * 3, h_dim=args.h_dim).to(device)
    checkpoint = torch.load("model/checkpoint_model.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model = nn.DataParallel(model).to(device)

    accr = test(model, test_loader)
    print(f"Accuracy: {accr * 100 :.2f}%")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
