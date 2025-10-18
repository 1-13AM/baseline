from typing import Optional
import torchvision
import pathlib
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import OrderedDict
import torch.nn as nn
import random

def temporal_subsample(first_idx: int, last_idx: int, strategy: str = 'uniform', num_output_frames: int = 16) -> np.ndarray:
    
    assert last_idx - first_idx + 1 >= num_output_frames, "num_frames must at least be equal to num_output_frames"
    if strategy == 'uniform':
        return np.linspace(first_idx, last_idx, num_output_frames, endpoint=True).astype(int)
    
    elif strategy == 'random':
        sampled_frame_idx = chunk_and_sample(first_idx, last_idx, num_output_frames)
        return np.array(sampled_frame_idx)
    
def chunk_and_sample(a, b, n):
    """
    Divides the sequence of numbers from 0 to n into k chunks, samples one value from each chunk,
    and returns a list of sampled values.
    
    Args:
        n (int): The end value of the sequence (exclusive), generating sequence from 0 to n-1.
        k (int): The number of chunks to divide the sequence into.
    
    Returns:
        list: A list of sampled values, one from each chunk.
    """
    boundaries = np.linspace(a, b, n + 1, endpoint=True).astype(int)
    
    sampled_values = []
    
    for i in range(0, len(boundaries) - 1):
        chunk = np.arange(boundaries[i], boundaries[i+1])
        if len(chunk) > 0:
            sampled_value = np.random.choice(chunk)
            sampled_values.append(sampled_value)
    
    return sampled_values


class RescaleTransform(torch.nn.Module):
    def forward(self, img):
        img = img * 2 - 1
        return img


def create_video_transforms(mean: tuple[float], std: tuple[float], resize_to: int, crop_size:  Optional[tuple[int,...]]) -> torch.nn.Sequential:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std),
        ShortSideScale(resize_to),
        transforms.CenterCrop(crop_size)
    ])

def create_video_transforms_i3d(resize_to: int, crop_size:  Optional[tuple[int,...]]) -> torch.nn.Sequential:
    return transforms.Compose([
        transforms.ToTensor(),
        RescaleTransform(),
        ShortSideScale(resize_to),
        transforms.CenterCrop(crop_size)
    ])
    
def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example['pixel_values'].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

class ShortSideScale(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        h, w = img.shape[-2:]
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
        else:
            oh = self.size
            ow = int(self.size * w / h)
        # torch.nn.functional.interpolate takes batched images as input
        if len(img.shape) == 3:
            img = img[None,...]
        return torch.nn.functional.interpolate(img, size=(oh, ow), mode="bilinear", align_corners=False)[0,:]            
    
def set_all_seeds(seed=42, deterministic=True):
    """
    Set all seeds to make results reproducible.
    
    Args:
        seed (int): Seed number, defaults to 42
        deterministic (bool): If True, ensures deterministic behavior in CUDA operations
                            Note that this may impact performance
    
    Note:
        Setting deterministic=True may significantly impact performance, but ensures
        complete reproducibility. If speed is crucial, you might want to set it to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False