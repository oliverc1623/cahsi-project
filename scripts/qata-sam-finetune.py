import argparse
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import PIL
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.nn.functional import threshold, normalize
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from transformers import SamProcessor
import datasets
from transformers import SamModel 
import loralib as lora
from tqdm import tqdm
from statistics import mean
import monai

numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def create_dataset(images, labels):
    print("Creating Dataset from dict")
    dataset = datasets.Dataset.from_dict({"image": images, "label": labels})
    return dataset

def process_data(image_file, mask=False):
    image = PIL.Image.open(image_file)
    if not mask:
        image = image.convert("RGB")
    else:
        image = image.convert("L")
        image = image.point(lambda p: p > 0 and 1)
    image = image.resize((256, 256), PIL.Image.BILINEAR)
    return image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def get_bounding_box(ground_truth_map):
    # get bounding box from mask
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

class SAMDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        prompt = get_bounding_box(ground_truth_mask)
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask
        return inputs

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
    model.mask_decoder.transformer.final_attn_token_to_image.q_proj = lora.Linear(256, 128, r=8)
    model.mask_decoder.transformer.final_attn_token_to_image.k_proj = lora.Linear(256, 128, r=8)
    model.mask_decoder.transformer.final_attn_token_to_image.v_proj = lora.Linear(256, 128, r=8)
    # Vision Encoder
    for layer in model.vision_encoder.layers:
        layer.attn.qkv = lora.MergedLinear(768, 3*768, r=8, enable_lora=[True, True, True])
        layer.mlp.lin1 = lora.Linear(768, 3072, r=8)
        layer.mlp.lin2 = lora.Linear(3072, 768, r=8)
    model.vision_encoder.neck.conv1 = lora.Conv2d(768, 256, kernel_size=1, r=8)
    model.vision_encoder.neck.conv2 = lora.Conv2d(256, 256, kernel_size=1, r=8)

def train_model(model, criterion, optimizer, train_dataloader, num_epochs=25, log_file_path="../../../pvcvolume/training_log.txt"):
    mean_epoch_losses = []
    mean_epoch_val_losses = []
    prev_val_loss = np.inf
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    model.train()
    
    with open(log_file_path, "w") as log_file:
        for epoch in range(num_epochs):
            epoch_losses = []
            # Training phase
            for batch in tqdm(train_dataloader):
                # forward pass
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                                  input_boxes=batch["input_boxes"].to(device),
                                  multimask_output=False)
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
           
            log_file.write(f"Epoch: {epoch}\t")
            log_file.write(f"Training loss: {mean_loss}\n")

            # save model if better
            if mean_loss < prev_val_loss:
                prev_val_loss = mean_loss
                torch.save(model.state_dict(), "../../../pvcvolume/baseline-sam-run.pth")
            model.train()

            print(f"Epoch: {epoch}")
            print(f"Training loss: {mean_loss}")

def main(subset_size, num_epochs):
    # Load raw data files
    train_filelist_xray = sorted(glob.glob('../QaTa-COV19/QaTa-COV19-v2/Train Set/Images/*.png'), key=numericalSort)
    x_train = [process_data(file_xray) for file_xray in train_filelist_xray[:subset_size]]
    masks = sorted(glob.glob('../QaTa-COV19/QaTa-COV19-v2/Train Set/Ground-truths/*.png'), key=numericalSort)
    y_train = [process_data(m, mask=True) for m in masks[:subset_size]]

    # create dictionary image, mask dataset
    dataset = create_dataset(x_train, y_train)
    print(dataset)

    # Fine-tuning
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Initialize Dataset and split into train and validation dataloaders
    dataset = SAMDataset(dataset=dataset, processor=processor)
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Load baseline model
    model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda:0")
    # only finetune vision encoder and mask decoder
    for name, param in model.named_parameters():
        if name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    sam_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SAM total params: {sam_total_params}")
    
    # train model
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    train_model(model, seg_loss, optimizer, train_dataloader, num_epochs=num_epochs)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAM Model")
    parser.add_argument("--subset_size", type=int, default=7145, help="Size of the dataset subset to use for training")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs for training")
    args = parser.parse_args()
    main(args.subset_size, args.num_epochs)
