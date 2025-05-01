from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import numpy as np
from utils import compute_z_order
from splitter import DensityBasedSplitter

class UniformPatchDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_patches,
        patch_size=16,
        is_train=False,
        image_resize=256,
        return_coords=True
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.image_paths = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for ext in ["*.jpeg", "*.jpg", "*.png"]:
                for img_path in class_dir.glob(ext):
                    self.image_paths.append((img_path, self.class_to_idx[class_name]))

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.is_train = is_train
        self.image_resize = image_resize
        self.return_coords = return_coords

        self.grid_size = self.image_resize // self.patch_size
        self.actual_num_patches = self.grid_size * self.grid_size
        
        if self.return_coords:
            self.coordinates_grid = self._create_coordinate_grid()

        if self.is_train:
            self.transforms = T.Compose([
                T.Resize((image_resize, image_resize)),
                T.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transforms = T.Resize((image_resize, image_resize))

        self.patch_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
    def _create_coordinate_grid(self):
        grid_h = grid_w = self.grid_size
        
        y_positions = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).expand(-1, grid_w).reshape(-1)
        x_positions = torch.arange(grid_w, dtype=torch.float32).expand(grid_h, -1).reshape(-1)
        
        y_positions = 2 * (y_positions / (grid_h - 1)) - 1 if grid_h > 1 else torch.zeros(1)
        x_positions = 2 * (x_positions / (grid_w - 1)) - 1 if grid_w > 1 else torch.zeros(1)
        
        return torch.stack([x_positions, y_positions], dim=1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        patches_with_coords = []
        for y in range(0, self.image_resize, self.patch_size):
            for x in range(0, self.image_resize, self.patch_size):
                patch = img.crop((x, y, x+self.patch_size, y+self.patch_size))
                patches_with_coords.append((patch, x, y))

        patches_with_z = []
        for patch, x, y in patches_with_coords:
            z = compute_z_order(x, y, self.image_resize, self.image_resize)
            patches_with_z.append((patch, z))
        patches_with_z.sort(key=lambda x: x[1])
        uniform_patches = [p[0] for p in patches_with_z]

        if len(uniform_patches) > self.num_patches:
            uniform_patches = uniform_patches[:self.num_patches]
        elif len(uniform_patches) < self.num_patches:
            needed = self.num_patches - len(uniform_patches)
            for _ in range(needed):
                uniform_patches.append(Image.new("RGB", (self.patch_size, self.patch_size), color=(0,0,0)))

        patch_tensors = [self.patch_normalize(p) for p in uniform_patches]
        patch_stack = torch.stack(patch_tensors, dim=0)
        patch_stack = patch_stack.view(self.num_patches, -1)
        
        if self.return_coords:
            coords = self.coordinates_grid[:self.num_patches]
            return patch_stack, coords, torch.tensor(label, dtype=torch.long)
        else:
            return patch_stack, torch.tensor(label, dtype=torch.long)

    @property
    def num_classes(self):
        return len(self.classes)


class DensityPatchDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_patches=194,
        patch_size=16,
        is_train=False,
        min_patch_size=16,
        blur_radius=2,
        canny_sigma=1.0,
        canny_low=50,
        canny_high=100,
        image_resize=256,
        min_variance_threshold=0.001,
        return_coords=True
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.image_paths = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for ext in ["*.jpeg", "*.jpg", "*.png"]:
                for img_path in class_dir.glob(ext):
                    self.image_paths.append((img_path, self.class_to_idx[class_name]))

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.is_train = is_train
        self.min_patch_size = min_patch_size
        self.blur_radius = blur_radius
        self.canny_sigma = canny_sigma
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.image_resize = image_resize
        self.min_variance_threshold = min_variance_threshold
        self.return_coords = return_coords

        if self.is_train:
            self.image_level_transform = T.Compose([
                T.Resize((image_resize, image_resize)),
                T.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.image_level_transform = T.Resize((image_resize, image_resize))

        self.patch_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        pil_img = Image.open(img_path).convert("RGB")
        pil_img = self.image_level_transform(pil_img)

        splitter = DensityBasedSplitter(
            pil_image=pil_img,
            target_count=self.num_patches,
            min_patch_size=self.min_patch_size,
            blur_radius=self.blur_radius,
            canny_sigma=self.canny_sigma,
            canny_low=self.canny_low,
            canny_high=self.canny_high
        )

        splitter.build_patches(min_variance_threshold=self.min_variance_threshold)
        if self.return_coords:
            patch_imgs, patch_coords = splitter.get_leaf_patches(
                patch_count=self.num_patches,
                patch_size=self.patch_size,
                z_order=True,
                drop_randomly=True
            )
        else:
            patch_imgs, _ = splitter.get_leaf_patches(
                patch_count=self.num_patches,
                patch_size=self.patch_size,
                z_order=True,
                drop_randomly=True
            )

        patch_tensors = [self.patch_normalize(patch) for patch in patch_imgs]
        patch_stack = torch.stack(patch_tensors, dim=0)
        patch_stack = patch_stack.view(self.num_patches, -1)
        
        if self.return_coords:
            coords_tensor = torch.tensor(patch_coords, dtype=torch.float32)
            return patch_stack, coords_tensor, torch.tensor(label, dtype=torch.long)
        else:
            return patch_stack, torch.tensor(label, dtype=torch.long)

    @property
    def num_classes(self):
        return len(self.classes)