# DO THIS ON YOUR FIRST RUN!

import torch
from typing import List
import argparse

def rename_layer_names(weight_path: str, new_weight_path: str, model_keys: List[str]):
    # Step 1: Load the .pth file (state dictionary)
    state_dict = torch.load(weight_path)

    # Step 2: Create a new dictionary with renamed keys
    new_state_dict = {}
    
    weight_keys = list(state_dict.keys())
    key_mapping = {
        weight_key: model_key for model_key, weight_key in zip(model_keys, weight_keys)
    }

    for old_key, value in state_dict.items():
        new_key = key_mapping.get(old_key, old_key)  # Use new key if mapped, otherwise keep old key
        new_state_dict[new_key] = value

    # Pop out classification head
    # for _ in range(2):
    #     new_state_dict.popitem()


    # Step 3: Save the modified state dictionary to a new .pth file
    torch.save(new_state_dict, new_weight_path)

if __name__ == "__main__":
    
    from models.VideoMAE_v2 import VideoMAEForClassification
    from functools import partial
    import torch.nn as nn
    
    parser = argparse.ArgumentParser(description='Align checkpoint layer names with model layer names')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='Checkpoint path')
    parser.add_argument('--new_weight_path', type=str, required=True,
                        help='New checkpoint path')
    args = parser.parse_args()

    # get model layer names
    model = VideoMAEForClassification(
                          num_classes = 400,
                          qkv_bias=True, 
                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          init_values=0.
                          )
    model_layer_names = []
    for name, param in model.named_parameters():
        model_layer_names.append(name)

    rename_layer_names(weight_path=args.weight_path,
                       new_weight_path=args.new_weight_path,
                       model_keys=model_layer_names)