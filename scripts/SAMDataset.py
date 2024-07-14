#%% 

import os
import glob
import warnings
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from transformers import SamProcessor
import numpy as np
warnings.filterwarnings('ignore')

#%%
def binarize_mask(mask):
    # Ensure the input is a tensor
    if not isinstance(mask, torch.Tensor):
        mask = transforms.ToTensor()(mask)
    # Apply the binary transformation and convert to integer
    return (mask > 0).int()

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

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path, processor, image_transform=None, mask_transform=None):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,"Images","*.png"))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,"Ground-truths",f"mask_{os.path.basename(img_path)}"))
        self.processor = processor
        self.transform = image_transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = Image.open(img_path).convert("RGB")
            label = Image.open(mask_path).convert("L")

            if self.transform:
                data = self.transform(data)
                
            if self.mask_transform:
                label = self.mask_transform(label).squeeze(0)

            prompt = get_bounding_box(np.array(label))
            inputs = self.processor(data, input_boxes=[[prompt]], return_tensors="pt", do_rescale=False)
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}
            inputs["ground_truth_mask"] = label
            return inputs

    def __len__(self):
        return len(self.img_files)
    
#%%

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(binarize_mask) 
])

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
                                         
ds = DataLoaderSegmentation(
    folder_path = "../QaTa-COV19/QaTa-COV19-v2/Train Set/",
    processor = processor,
    image_transform = transform,
    mask_transform = mask_transform
)
