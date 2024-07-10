import os
import argparse
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from data import HAM10000, train_test_split

from model import ResNet50
from tqdm import tqdm

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--lr", default=1e-4, type=float)

parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--use_wandb", default=False, type=bool)

IMAGE_DIR = "dataset/imgs"
LABEL_PATH = "dataset/label.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def init_weight(model):
    cname = model.__class__.__name__
    if cname.find("Linear") != -1:
        nn.init.kaiming_normal_(model.weight)


def validate(model, data_loader, loss_fn):
    model.eval()
    avg_loss, accr = 0, 0

    with torch.no_grad():
        with tqdm(data_loader, desc="Validation") as pbar:
            for X, y in pbar:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                avg_loss += loss_fn(pred, y).item()
                accr += (pred.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()

    avg_loss /= len(data_loader)
    accr /= len(data_loader.dataset)
    return avg_loss, accr


def train(args, model, train_loader, val_loader, optim, loss_fn, scheduler):
    best_loss, best_model = 0, model
    for epoch in range(args.epochs):
        train_loss, val_loss = 0, 0
        model.train()
        with tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.epochs}") as pbar:
            for X, y in pbar:
                X, y = X.to(device), y.to(device)
                breakpoint()
                pred = model(X)
                loss = loss_fn(pred, y)
                train_loss += loss.item()
                loss.backward()
                optim.step()
                optim.zero_grad()
        train_loss /= len(train_loader)
        scheduler.step()

        val_loss, accr = validate(model, val_loader, loss_fn)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

        if args.use_wandb:
            wandb.log(
                {
                    "lr": optim.param_groups[0]["lr"],
                    "train loss": train_loss,
                    "val loss": val_loss,
                    "accuracy": accr,
                }
            )

    return best_model


def main(args):
    if args.use_wandb:
        wandb.init(project="ham_classification")

    dataset = HAM10000(
        image_dir=IMAGE_DIR,
        label_path=LABEL_PATH,
    )
    train_data, val_data = train_test_split(dataset)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
    )

    model = ResNet50()
    model = nn.DataParallel(model).to(device)
    # model.apply(init_weight)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-8)

    best_model = train(args, model, train_loader, val_loader, optim, loss_fn, scheduler)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)