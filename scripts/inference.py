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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from transformers import SamProcessor
import datasets
from transformers import SamModel 
import loralib as lora
from tqdm import tqdm
from statistics import mean
import monai
numbers = re.compile(r'(\d+)')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def create_dataset(images, labels):
    dataset = datasets.Dataset.from_dict({"image": images,
                                "label": labels})
    dataset = dataset.cast_column("image", datasets.Image())
    dataset = dataset.cast_column("label", datasets.Image())
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
        # Calculate the true positives,
        # false positives, and false negatives
        tp = 0
        fp = 0
        fn = 0
        for i in range(len(gtMask)):
            for j in range(len(gtMask[0])):
                if gtMask[i][j] == 1 and predMask[i][j] == 1:
                    tp += 1
                elif gtMask[i][j] == 0 and predMask[i][j] == 1:
                    fp += 1
                elif gtMask[i][j] == 1 and predMask[i][j] == 0:
                    fn += 1
        # Calculate IoU
        iou = tp / (tp + fp + fn)
        return iou

def main():
    ## Load Test Set
    subset_size = 100
    test_filelist_xray = sorted(glob.glob('../QaTa-COV19/QaTa-COV19-v2/Test Set/Images/*.png'))
    x_test = np.array([process_data(file_xray) for file_xray in test_filelist_xray[:subset_size]])
    test_masks = sorted(glob.glob('../QaTa-COV19/QaTa-COV19-v2/Test Set/Ground-truths/*.png'))
    y_test = np.array([process_data(m, mask=True) for m in test_masks[:subset_size]])
    y_test[y_test > 0] = 1 
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels (masks) data shape: {y_test.shape}")
    test_dataset = create_dataset(x_test, y_test)

    # Load model
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamProcessor.from_pretrained("/home/../pvcvolume/sam_checkpoints/")
    
    # Inference on test set
    test_ious = []
    for idx, sample in enumerate(test_dataset):
        ground_truth_mask = np.array(sample["label"])
        prompt = get_bounding_box(ground_truth_mask)        
        image = sample["image"]
        inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt").to(device)
        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        iou = calculateIoU(ground_truth_mask, medsam_seg)
        test_ious.append(iou)
    
    print(f"Average IoUs over {subset_size} test samples: {mean(test_ious)}")

if __name__ == "__main__":
    main()
