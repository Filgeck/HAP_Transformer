import os
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import sys

sys.path.append('.')
from inference import parse_model_name, load_model, process_image_uniform, process_image_density, predict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_in_memory(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024  # in MB

def test_model_efficiency(model_path, test_images_dir, class_names):
    print(f"Loading model from {model_path}")
    try:
        config = parse_model_name(model_path)
        model, device = load_model(model_path, config)
        model_params = count_parameters(model)
        model_memory = get_model_size_in_memory(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {
            'model_name': os.path.basename(model_path),
            'error': f"Failed to load model: {str(e)}",
            'parameters': 0,
            'memory_mb': 0,
            'avg_patch_time': float('nan'),
            'avg_model_time': float('nan'),
            'avg_total_time': float('nan')
        }

    patch_times = []
    model_times = []
    total_times = []

    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    test_image_paths = [os.path.join(test_images_dir, f) for f in test_images]
    
    if not test_image_paths:
        print(f"No images found in {test_images_dir}. Please add some test images.")
        return {
            'model_name': os.path.basename(model_path),
            'config': config,
            'parameters': model_params,
            'memory_mb': model_memory,
            'avg_patch_time': float('nan'),
            'avg_model_time': float('nan'),
            'avg_total_time': float('nan')
        }
    
    print(f"Processing {len(test_image_paths)} test images...")
    
    for i, image_path in enumerate(test_image_paths):
        print(f"  Processing image {i+1}/{len(test_image_paths)}: {os.path.basename(image_path)}")
        try:
            total_start_time = time.time()
            
            patch_start_time = time.time()
            
            if config['patch_method'] == 'uniform':
                patches, coords = process_image_uniform(image_path, config)
            else:
                patches, coords = process_image_density(image_path, config)

            patch_end_time = time.time()
            patch_time = patch_end_time - patch_start_time
            patch_times.append(patch_time)

            model_start_time = time.time()
            
            pred_class, pred_idx, confidence = predict(model, patches, coords, device, class_names)

            model_end_time = time.time()
            model_time = model_end_time - model_start_time
            model_times.append(model_time)

            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            total_times.append(total_time)
            
            print(f"    Prediction: {pred_class} (Confidence: {confidence:.4f})")
            print(f"    Times: Patch={patch_time:.4f}s, Model={model_time:.4f}s, Total={total_time:.4f}s")
            
        except Exception as e:
            print(f"    Error processing image {image_path}: {e}")

    if patch_times:
        avg_patch_time = np.mean(patch_times)
        avg_model_time = np.mean(model_times)
        avg_total_time = np.mean(total_times)
    else:
        avg_patch_time = float('nan')
        avg_model_time = float('nan')
        avg_total_time = float('nan')
    
    return {
        'model_name': os.path.basename(model_path),
        'config': config,
        'parameters': model_params,
        'memory_mb': model_memory,
        'avg_patch_time': avg_patch_time,
        'avg_model_time': avg_model_time,
        'avg_total_time': avg_total_time,
        'num_images_processed': len(patch_times)
    }

def main():
    parser = argparse.ArgumentParser(description='Test efficiency of Vision Transformer models')
    parser.add_argument('--test_dir', type=str, default='./test_images', 
                        help='Directory containing test images')
    parser.add_argument('--classes', type=str, default='./classes.txt',
                        help='Path to class names file')
    parser.add_argument('--models_dir', type=str, default='./pretrained',
                        help='Directory containing model files')
    args = parser.parse_args()

    if not os.path.exists(args.test_dir):
        print(f"Test directory {args.test_dir} does not exist. Creating it...")
        os.makedirs(args.test_dir)
        print(f"Please add some test images to {args.test_dir} and run the script again.")
        return

    if not os.path.exists(args.classes):
        print(f"Class names file {args.classes} does not exist.")
        print("Creating a sample classes.txt file with 10 classes...")
        with open(args.classes, 'w') as f:
            for i in range(10):
                f.write(f"class_{i}\n")

    try:
        with open(args.classes, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading class names: {e}")
        class_names = [f"class_{i}" for i in range(10)]

    model_files = [
        "vit_uniform_learned_256_best.pth",
        "vit_uniform_learned_256_best.pth",
        "vit_density_coordinate_130_best.pth", 
        "vit_density_coordinate_160_best.pth",
        "vit_density_coordinate_202_best.pth"
    ]
    
    models = [os.path.join(args.models_dir, model) for model in model_files]

    existing_models = []
    for model_path in models:
        if os.path.exists(model_path):
            existing_models.append(model_path)
        else:
            print(f"Warning: Model {model_path} does not exist and will be skipped.")
    
    if not existing_models:
        print(f"No models found in {args.models_dir}. Please add models and run again.")
        return

    results = []
    for model_path in existing_models:
        print(f"\nTesting model: {model_path}")
        result = test_model_efficiency(model_path, args.test_dir, class_names)
        results.append(result)

    print("\n" + "="*80)
    print("MODEL EFFICIENCY COMPARISON")
    print("="*80)
    
    print("\nMODEL PARAMETERS:")
    for result in results:
        model_name = result['model_name']
        if 'error' in result:
            print(f"{model_name}: {result['error']}")
            continue
            
        config = result['config']
        params = result['parameters']
        memory_mb = result['memory_mb']
        
        print(f"{model_name}:")
        print(f"  - Patch Method: {config['patch_method']}")
        print(f"  - Position Embedding: {config['pos_embed_type']}")
        print(f"  - Number of Patches: {config['num_patches']}")
        print(f"  - Parameters: {params:,} (~{memory_mb:.2f} MB)")
        if 'num_images_processed' in result:
            print(f"  - Images processed: {result['num_images_processed']}")
        print()

    valid_results = [r for r in results if not np.isnan(r['avg_total_time'])]
    
    if valid_results:
        print("\nAVERAGE TIMING RESULTS (seconds):")
        print(f"{'Model':<40} {'Patch Creation':>15} {'Model Inference':>15} {'Total Time':>15}")
        print("-"*90)
        
        for result in valid_results:
            model_name = result['model_name']
            avg_patch = result['avg_patch_time']
            avg_model = result['avg_model_time']
            avg_total = result['avg_total_time']
            
            print(f"{model_name:<40} {avg_patch:>15.4f} {avg_model:>15.4f} {avg_total:>15.4f}")

        fastest_patch = min(valid_results, key=lambda x: x['avg_patch_time'])
        fastest_model = min(valid_results, key=lambda x: x['avg_model_time'])
        fastest_total = min(valid_results, key=lambda x: x['avg_total_time'])
        
        print("\nFASTEST MODELS:")
        print(f"Fastest Patch Creation: {fastest_patch['model_name']} ({fastest_patch['avg_patch_time']:.4f}s)")
        print(f"Fastest Model Inference: {fastest_model['model_name']} ({fastest_model['avg_model_time']:.4f}s)")
        print(f"Fastest Total Inference: {fastest_total['model_name']} ({fastest_total['avg_total_time']:.4f}s)")
    else:
        print("\nNo valid timing results available. Please check that test images are properly processed.")

if __name__ == "__main__":
    main()