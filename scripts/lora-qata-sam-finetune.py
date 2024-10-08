import argparse
import glob
import os
import re
import warnings
from statistics import mean

import loralib as lora
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


def apply_lora(model):
    # Mask decoder
    for layer in model.mask_decoder.transformer.layers:
        # self_attn
        layer.self_attn.q_proj = lora.Linear(256, 256, r=8)
        layer.self_attn.k_proj = lora.Linear(256, 256, r=8)
        layer.self_attn.v_proj = lora.Linear(256, 256, r=8)
        # cross attn token to img
        layer.cross_attn_token_to_image.q_proj = lora.Linear(256, 128, r=8)
        layer.cross_attn_token_to_image.k_proj = lora.Linear(256, 128, r=8)
        layer.cross_attn_token_to_image.v_proj = lora.Linear(256, 128, r=8)
        # mlp
        layer.mlp.lin1 = lora.Linear(256, 2048, r=8)
        layer.mlp.lin2 = lora.Linear(2048, 256, r=8)
        # cross attn img to token
        layer.cross_attn_image_to_token.q_proj = lora.Linear(256, 128, r=8)
        layer.cross_attn_image_to_token.k_proj = lora.Linear(256, 128, r=8)
        layer.cross_attn_image_to_token.v_proj = lora.Linear(256, 128, r=8)
    model.mask_decoder.transformer.final_attn_token_to_image.q_proj = lora.Linear(
        256, 128, r=8
    )
    model.mask_decoder.transformer.final_attn_token_to_image.k_proj = lora.Linear(
        256, 128, r=8
    )
    model.mask_decoder.transformer.final_attn_token_to_image.v_proj = lora.Linear(
        256, 128, r=8
    )
    # Vision Encoder
    for layer in model.vision_encoder.layers:
        layer.attn.qkv = lora.MergedLinear(
            768, 3 * 768, r=8, enable_lora=[True, True, True]
        )
        layer.mlp.lin1 = lora.Linear(768, 3072, r=8)
        layer.mlp.lin2 = lora.Linear(3072, 768, r=8)
    model.vision_encoder.neck.conv1 = lora.Conv2d(768, 256, kernel_size=1, r=8)
    model.vision_encoder.neck.conv2 = lora.Conv2d(256, 256, kernel_size=1, r=8)


def train_model(model, criterion, optimizer, train_dataloader, num_epochs=25):
    mean_epoch_losses = []
    prev_val_loss = np.inf

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_losses = []
        # Training phase
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False,
            )
            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = criterion(predicted_masks, ground_truth_masks.unsqueeze(1))
            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()
            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        # log statistics
        mean_loss = mean(epoch_losses)
        mean_epoch_losses.append(mean_loss)

        # save model if better
        if mean_loss < prev_val_loss:
            prev_val_loss = mean_loss
            torch.save(model.state_dict(), "../../pvcvolume/lora-sam-run.pth")
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

    # Fine-tuning
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Initialize Dataset and split into train and validation dataloaders
    sam_dataset = SAMDataset(
        folder_path="../../pvcvolume/QaTa-COV19/QaTa-COV19-v2/Train Set/",
        processor=processor,
        image_transform=transform,
        mask_transform=mask_transform,
    )
    sam_dataset = Subset(sam_dataset, subset_indices)

    # Initialize Dataset and split into train and validation dataloaders
    train_dataloader = DataLoader(
        sam_dataset, batch_size=8, shuffle=True, num_workers=4
    )

    # Load baseline model
    model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda:0")

    # only finetune vision encoder and mask decoder
    for name, param in model.named_parameters():
        if name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    sam_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SAM total params: {sam_total_params}")

    apply_lora(model)
    sam_lora_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"SAM + LoRA total params: {sam_lora_total_params}")

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
