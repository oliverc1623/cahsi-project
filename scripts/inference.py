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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")


def get_bounding_box(ground_truth_map):
    # get bounding box from mask
    if len(ground_truth_map) == 2:
        ground_truth_map = ground_truth_map[0]
    y_indices, x_indices = np.where(ground_truth_map > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]
    return bbox


def calculateIoU(gtMask, predMask):
    intersection = torch.sum(predMask * gtMask)
    union = torch.sum(predMask) + torch.sum(gtMask) - intersection
    if union == 0:
        iou = float("nan")
    else:
        iou = intersection / union
    return iou.item()


def main(subset_size):
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
        folder_path="../../pvcvolume/QaTa-COV19/QaTa-COV19-v2/Test Set/",
        processor=processor,
        image_transform=transform,
        mask_transform=mask_transform,
    )
    sam_dataset = Subset(sam_dataset, subset_indices)

    test_dataloader = DataLoader(sam_dataset, batch_size=8, shuffle=True, num_workers=4)

    # Load baseline model
    model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda:0")
    # only finetune vision encoder and mask decoder
    for name, param in model.named_parameters():
        if name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    sam_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SAM total params: {sam_total_params}")

    model.load_state_dict(torch.load("../../pvcvolume/baseline-sam-run.pth"))

    test_ious = []
    model.eval()
    with torch.no_grad():
        for indx, batch in enumerate(
            test_dataloader
        ):  # Make sure to use your validation DataLoader
            # forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False,
            )
            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            medsam_seg_prob = torch.sigmoid(predicted_masks)
            medsam_seg_prob = medsam_seg_prob.squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).to(torch.uint8)
            for i in range(len(medsam_seg)):
                iou = calculateIoU(ground_truth_masks[i], medsam_seg[i])
                test_ious.append(iou)
                print(f"Test sample index: {indx}, i: {i}, iou: {iou}")

    print(f"Average IoUs over {subset_size} test samples: {mean(test_ious)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM Model")
    parser.add_argument(
        "--subset_size",
        type=int,
        default=2113,
        help="Size of the dataset subset to use for training",
    )
    args = parser.parse_args()
    main(args.subset_size)
