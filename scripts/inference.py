import re
import glob
import numpy as np
import PIL.Image
import PIL
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.nn.functional import threshold, normalize
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from transformers import SamModel, SamConfig, SamProcessor
import datasets
import loralib as lora
from tqdm import tqdm
from statistics import mean
import monai
device = "cuda" if torch.cuda.is_available() else "cpu"

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def create_dataset(images, labels):
    print("Creating Dataset from dict")
    dataset = datasets.Dataset.from_dict({"image": images,
                                "label": labels})
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
        iou = float('nan')
    else:
        iou = intersection / union
    return iou.item()

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

def main():
    ## Load Test Set
    subset_size = 400
    test_filelist_xray = sorted(glob.glob('../QaTa-COV19/QaTa-COV19-v2/Test Set/Images/*.png'))
    x_test = [process_data(file_xray) for file_xray in test_filelist_xray[:subset_size]]

    test_masks = sorted(glob.glob('../QaTa-COV19/QaTa-COV19-v2/Test Set/Ground-truths/*.png'))
    y_test = [process_data(m, mask=True) for m in test_masks[:subset_size]]

    test_dataset = create_dataset(x_test, y_test)

    # Load model
    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base") 
    model = SamModel(config=model_config)
    apply_lora(model)
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    model.load_state_dict(torch.load("../lora-sam.pth"))

    # Initialize Dataset and split into train and validation dataloaders
    test_dataset = SAMDataset(dataset=test_dataset, processor=processor)
    dataset_size = len(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=8)

    test_ious = []
    model.eval()
    with torch.no_grad():
        for indx, batch in enumerate(test_dataloader):  # Make sure to use your validation DataLoader
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)
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
    main()
