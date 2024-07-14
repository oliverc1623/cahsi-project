import argparse
import glob
import os
import re
import warnings
from statistics import mean

import monai
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from SAMDataset import SAMDataset, binarize_mask
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import SamModel, SamProcessor

warnings.filterwarnings("ignore")


def train_model(model, criterion, optimizer, train_dataloader, num_epochs=25):
    print(f"Beginning training for {num_epochs} epochs")
    mean_epoch_losses = []
    prev_val_loss = np.inf

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)
    model.train()
    print("Model loaded on device")

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        epoch_losses = []
        # Training phase
        for batch in tqdm(train_dataloader):
            # forward pass
            print(batch["pixel_values"].shape)
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False,
            )
            print("passed forward pass")
            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = criterion(predicted_masks, ground_truth_masks.unsqueeze(1))
            # backward pass (compute gradients of parameters w.r.t. loss)
            print("computing gradients")
            optimizer.zero_grad()
            loss.backward()
            # optimize
            optimizer.step()
            print("optimizing")
            epoch_losses.append(loss.item())

        # log statistics
        mean_loss = mean(epoch_losses)
        mean_epoch_losses.append(mean_loss)

        # save model if better
        if mean_loss < prev_val_loss:
            print(f"Saving model with loss: {mean_loss}")
            prev_val_loss = mean_loss
            torch.save(model.state_dict(), "../../pvcvolume/baseline-sam-run.pth")
        model.train()

        print(f"Epoch: {epoch}")
        print(f"Training loss: {mean_loss}")


def main(subset_size, num_epochs):
    subset_indices = list(range(subset_size))

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    mask_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(binarize_mask),
        ]
    )

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    sam_dataset = SAMDataset(
        folder_path="../../pvcvolume/QaTa-COV19/QaTa-COV19-v2/Train Set/",
        processor=processor,
        image_transform=transform,
        mask_transform=mask_transform,
    )
    sam_dataset = Subset(sam_dataset, subset_indices)

    # Initialize Dataset and split into train and validation dataloaders
    train_dataloader = DataLoader(
        sam_dataset, batch_size=4, shuffle=True, num_workers=4
    )

    # Load baseline model
    model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda:0")

    # only finetune vision encoder and mask decoder
    for name, param in model.named_parameters():
        if name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    sam_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SAM total params: {sam_total_params}")

    # train model
    optimizer = Adam(model.parameters(), lr=4e-5, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(
        sigmoid=True, squared_pred=True, reduction="mean"
    )
    train_model(model, seg_loss, optimizer, train_dataloader, num_epochs=num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM Model")
    parser.add_argument(
        "--subset_size",
        type=int,
        default=7145,
        help="Size of the dataset subset to use for training",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=25, help="Number of epochs for training"
    )
    args = parser.parse_args()
    main(args.subset_size, args.num_epochs)
