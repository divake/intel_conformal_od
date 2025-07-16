#!/usr/bin/env python3
"""
Inspect the Faster R-CNN model checkpoint
"""
import torch
import pickle

model_path = "checkpoints/faster_rcnn_X_101_32x8d_FPN_3x.pth"

print(f"Inspecting: {model_path}")
print("=" * 50)

# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

# Check type
print(f"Type: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print(f"\nKeys in checkpoint:")
    for i, key in enumerate(list(checkpoint.keys())[:10]):
        print(f"  - {key}")
        if key == 'model':
            print(f"    Type: {type(checkpoint[key])}")
            if isinstance(checkpoint[key], dict):
                print(f"    Model keys: {list(checkpoint[key].keys())[:5]}...")
    
    if len(checkpoint.keys()) > 10:
        print(f"  ... and {len(checkpoint.keys()) - 10} more keys")
else:
    print("Checkpoint is not a dictionary")

# Check for specific keys
important_keys = ['model', 'state_dict', 'optimizer', 'iteration', 'cfg']
print(f"\nChecking for important keys:")
for key in important_keys:
    if isinstance(checkpoint, dict) and key in checkpoint:
        print(f"  ✓ {key} found")