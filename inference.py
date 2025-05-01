import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import os
from pathlib import Path
import numpy as np
import re
import time

from model import VisionTransformer
from splitter import DensityBasedSplitter
from utils import compute_z_order

def parse_model_name(model_path):
    filename = os.path.basename(model_path)
    pattern = r"vit_(\w+)_(\w+)_(\d+)_best\.pth"
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Invalid model filename format: {filename}. Expected format: vit_[patch_method]_[pos_embed_type]_[patch_num]_best.pth")
    
    patch_method, pos_embed_type, patch_num = match.groups()
    patch_num = int(patch_num)
    
    config = {
        'patch_method': patch_method,
        'pos_embed_type': pos_embed_type,
        'num_patches': patch_num,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 6,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'num_classes': 10,
        'image_resize': 256,
        'min_patch_size': 16,
        'blur_radius': 2,
        'canny_sigma': 1.0,
        'canny_low': 30,
        'canny_high': 100,
        'min_variance_threshold': 0.001
    }
    
    return config

def load_model(model_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VisionTransformer(
        patch_dim=3 * config['patch_size'] * config['patch_size'],
        max_patches=config['num_patches'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        num_classes=config['num_classes'],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        pos_embed_type=config['pos_embed_type'],
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

def process_image_uniform(image_path, config):
    img = Image.open(image_path).convert("RGB")
    transform = T.Resize((config['image_resize'], config['image_resize']))
    img = transform(img)

    patches_with_coords = []
    for y in range(0, config['image_resize'], config['patch_size']):
        for x in range(0, config['image_resize'], config['patch_size']):
            patch = img.crop((x, y, x+config['patch_size'], y+config['patch_size']))
            patches_with_coords.append((patch, x, y))

    patches_with_z = []
    for patch, x, y in patches_with_coords:
        z = compute_z_order(x, y, config['image_resize'], config['image_resize'])
        patches_with_z.append((patch, z, x, y))
    patches_with_z.sort(key=lambda x: x[1])

    uniform_patches = [p[0] for p in patches_with_z[:config['num_patches']]]
    
    if len(uniform_patches) < config['num_patches']:
        needed = config['num_patches'] - len(uniform_patches)
        for _ in range(needed):
            uniform_patches.append(Image.new("RGB", (config['patch_size'], config['patch_size']), color=(0,0,0)))
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    patch_tensors = [normalize(p) for p in uniform_patches]
    patch_stack = torch.stack(patch_tensors, dim=0)
    patch_stack = patch_stack.view(config['num_patches'], -1)
    
    if config['pos_embed_type'] == 'coordinate':
        grid_size = config['image_resize'] // config['patch_size']
        coords = []
        for i, (_, _, x, y) in enumerate(patches_with_z[:config['num_patches']]):
            norm_x = 2 * (x / config['image_resize'] + 0.5 / grid_size) - 1
            norm_y = 2 * (y / config['image_resize'] + 0.5 / grid_size) - 1
            coords.append((norm_x, norm_y))

        if len(coords) < config['num_patches']:
            needed = config['num_patches'] - len(coords)
            for _ in range(needed):
                coords.append((0.0, 0.0))
        
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        return patch_stack.unsqueeze(0), coords_tensor.unsqueeze(0)
    else:
        return patch_stack.unsqueeze(0), None

def process_image_density(image_path, config):
    img = Image.open(image_path).convert("RGB")
    transform = T.Resize((config['image_resize'], config['image_resize']))
    img = transform(img)

    splitter = DensityBasedSplitter(
        pil_image=img,
        target_count=config['num_patches'],
        min_patch_size=config['min_patch_size'],
        blur_radius=config['blur_radius'],
        canny_sigma=config['canny_sigma'],
        canny_low=config['canny_low'],
        canny_high=config['canny_high']
    )
    
    splitter.build_patches(min_variance_threshold=config['min_variance_threshold'])
    patch_imgs, patch_coords = splitter.get_leaf_patches(
        patch_count=config['num_patches'],
        patch_size=config['patch_size'],
        z_order=True,
        drop_randomly=True
    )
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    patch_tensors = [normalize(p) for p in patch_imgs]
    patch_stack = torch.stack(patch_tensors, dim=0)
    patch_stack = patch_stack.view(config['num_patches'], -1)
    
    coords_tensor = torch.tensor(patch_coords, dtype=torch.float32)
    
    return patch_stack.unsqueeze(0), coords_tensor.unsqueeze(0)

def predict(model, patches, coords, device, class_names):
    patches = patches.to(device)
    
    with torch.no_grad():
        if coords is not None:
            coords = coords.to(device)
            logits = model(patches, patch_coords=coords)
        else:
            logits = model(patches)

        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        
        pred_class = class_names[pred_idx.item()]
        confidence = confidence.item()
        
    return pred_class, pred_idx.item(), confidence

def main():
    parser = argparse.ArgumentParser(description='Run inference on an image using a trained model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--classes', type=str, required=True, help='Path to class names file')
    args = parser.parse_args()

    total_start_time = time.time()

    config = parse_model_name(args.model)
    
    print(f"Model configuration:")
    print(f"  Patch method: {config['patch_method']}")
    print(f"  Position embedding type: {config['pos_embed_type']}")
    print(f"  Number of patches: {config['num_patches']}")
    
    with open(args.classes, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    model, device = load_model(args.model, config)
    
    patch_start_time = time.time()
    
    if config['patch_method'] == 'uniform':
        patches, coords = process_image_uniform(args.image, config)
    else:
        patches, coords = process_image_density(args.image, config)
    
    patch_end_time = time.time()
    patch_time = patch_end_time - patch_start_time
    
    model_start_time = time.time()
    
    pred_class, pred_idx, confidence = predict(model, patches, coords, device, class_names)
    
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"Prediction: {pred_class} (index: {pred_idx})")
    print(f"Confidence: {confidence:.4f}")
    
    print("\nInference Timing:")
    print(f"1) Creating patches: {patch_time:.4f} seconds")
    print(f"2) Running model:    {model_time:.4f} seconds")
    print(f"3) Total time:       {total_time:.4f} seconds")
    
    return pred_class, confidence

if __name__ == "__main__":
    main()