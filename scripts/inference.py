import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import PIL
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import SamProcessor
from datasets import Dataset, Image, load_dataset, Features, Array3D, ClassLabel
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

def process_data(image_file, mask=False):
    image = PIL.Image.open(image_file)
    if not mask:
        image = image.convert("RGB")
    else:
        image = image.convert("L")
        image = image.point(lambda p: p > 0 and 1)
    image = image.resize((256, 256), PIL.Image.BILINEAR)
    return image
    
def create_dataset(images, labels):
    dataset = Dataset.from_dict({"image": images,
                                "label": labels})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())
    return dataset

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
    model = SamProcessor.from_pretrained("/home/../pvcvolume/sam_checkpoints")
    
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