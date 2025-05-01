import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from datasets import UniformPatchDataset, DensityPatchDataset
from model import VisionTransformer

def train_one_epoch(model, dataloader, criterion, optimizer, device, use_coords=False):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        if use_coords:
            patches, coords, labels = batch
            patches = patches.to(device)
            coords = coords.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(patches, patch_coords=coords)
        else:
            if len(batch) == 3:
                patches, _, labels = batch
            else:
                patches, labels = batch
            
            patches = patches.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(patches)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

        total_loss += loss.item() * patches.size(0)
        total_samples += patches.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def validate(model, dataloader, criterion, device, use_coords=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            if use_coords:
                patches, coords, labels = batch
                patches = patches.to(device)
                coords = coords.to(device)
                labels = labels.to(device)
                
                logits = model(patches, patch_coords=coords)
            else:
                if len(batch) == 3:
                    patches, _, labels = batch
                else:
                    patches, labels = batch
                
                patches = patches.to(device)
                labels = labels.to(device)
                
                logits = model(patches)
            
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * patches.size(0)
            total_samples += patches.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def train_model(p_method="uniform", embed_type="learned", patch_num=256, epochs_num=30):
    patch_method = p_method
    pos_embed_type = embed_type
    use_coords = (pos_embed_type == 'coordinate')
    num_patches = patch_num
    patch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Patch method: {patch_method}")
    print(f"Position embedding type: {pos_embed_type}")
    print(f"Using coordinates: {use_coords}")
    print(f"Number of patches: {num_patches}")
    print(f"Patch size: {patch_size}")

    embed_dim = 384
    depth = 6
    num_heads = 6
    mlp_ratio = 4.0
    drop_rate = 0.1
    attn_drop_rate = 0.1
    num_epochs = epochs_num
    batch_size = 64
    learning_rate = 5e-4
    weight_decay = 1e-4
    min_patch_size = 16
    image_resize = 256

    min_variance_threshold = 0.001
    canny_low = 30
    canny_high = 100

    train_dir = "./imagenette/train"
    val_dir   = "./imagenette/val"

    if patch_method == 'uniform':
        print("Using uniform grid patches dataset...")
        train_dataset = UniformPatchDataset(
            root_dir=train_dir,
            num_patches=num_patches,
            patch_size=patch_size,
            is_train=True,
            image_resize=image_resize,
            return_coords=use_coords
        )
        val_dataset = UniformPatchDataset(
            root_dir=val_dir,
            num_patches=num_patches,
            patch_size=patch_size,
            is_train=False,
            image_resize=image_resize,
            return_coords=use_coords
        )
    else:
        print("Using Density-based patches dataset...")
        train_dataset = DensityPatchDataset(
            root_dir=train_dir,
            num_patches=num_patches,
            patch_size=patch_size,
            is_train=True,
            min_patch_size=min_patch_size,
            blur_radius=2,
            canny_sigma=1.0,
            canny_low=canny_low,
            canny_high=canny_high,
            image_resize=image_resize,
            min_variance_threshold=min_variance_threshold,
            return_coords=use_coords
        )
        val_dataset = DensityPatchDataset(
            root_dir=val_dir,
            num_patches=num_patches,
            patch_size=patch_size,
            is_train=False,
            min_patch_size=min_patch_size,
            blur_radius=2,
            canny_sigma=1.0,
            canny_low=canny_low,
            canny_high=canny_high,
            image_resize=image_resize,
            min_variance_threshold=min_variance_threshold,
            return_coords=use_coords
        )

    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VisionTransformer(
        patch_dim=3*patch_size*patch_size,
        max_patches=num_patches,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        pos_embed_type=pos_embed_type,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_acc = 0.0
    model_save_path = f"vit_{patch_method}_{pos_embed_type}_{num_patches}_best.pth"
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epochs = list(range(1, num_epochs + 1))

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_coords)
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_coords)
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | Time: {elapsed:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"** Saved new best model (val_acc={val_acc:.4f})")

    model.load_state_dict(torch.load(model_save_path))
    final_val_loss, final_val_acc = validate(model, val_loader, criterion, device, use_coords)
    print(f"\nFinal Model ({patch_method} approach with {pos_embed_type} positional embedding): Val Loss: {final_val_loss:.4f}, Val Acc: {final_val_acc:.4f}")
    
    return {
        'model': model,
        'model_name': f"{patch_method}_{pos_embed_type}_{num_patches}",
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'final_val_acc': final_val_acc,
        'final_val_loss': final_val_loss,
        'epochs': epochs
    }

def plot_model_comparisons(model_results, metrics=('loss', 'accuracy'), save_path='model_comparison.png'):
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'x', '*']
    
    plt.figure(figsize=(15, 10))
    
    if 'loss' in metrics:
        plt.subplot(2, 1, 1)
        for i, result in enumerate(model_results):
            model_name = result['model_name']
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            plot_every = max(1, len(result['epochs']) // 20)
            
            plt.plot(result['epochs'], result['train_losses'], 
                     linestyle='--', color=color, marker=marker, markevery=plot_every,
                     label=f"{model_name} (Train)")
            plt.plot(result['epochs'], result['val_losses'], 
                     linestyle='-', color=color, marker=marker, markevery=plot_every,
                     label=f"{model_name} (Validation)")
        
        plt.title('Loss Comparison', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    if 'accuracy' in metrics:
        plt.subplot(2, 1, 2 if 'loss' in metrics else 1)
        for i, result in enumerate(model_results):
            model_name = result['model_name']
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            plot_every = max(1, len(result['epochs']) // 20)
            
            plt.plot(result['epochs'], result['train_accs'], 
                     linestyle='--', color=color, marker=marker, markevery=plot_every,
                     label=f"{model_name} (Train)")
            plt.plot(result['epochs'], result['val_accs'], 
                     linestyle='-', color=color, marker=marker, markevery=plot_every,
                     label=f"{model_name} (Validation)")
        
        plt.title('Accuracy Comparison', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n--- Model Performance Summary ---")
    for result in model_results:
        print(f"{result['model_name']}: Final Val Loss = {result['final_val_loss']:.4f}, Val Acc = {result['final_val_acc']:.4f}")
    
    return save_path