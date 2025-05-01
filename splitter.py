import numpy as np
import random
from PIL import ImageFilter
from skimage.feature import canny
import heapq
from utils import compute_z_order
from PIL import Image, ImageDraw, ImageFilter

class DensityBasedSplitter:
    def __init__(self, pil_image, target_count, min_patch_size=16,
                 blur_radius=2, canny_sigma=1.0, canny_low=50, canny_high=100):
        self.original_image = pil_image
        self.width, self.height = pil_image.size
        self.target_count = target_count
        self.min_patch_size = min_patch_size

        gray = pil_image.convert("L")
        if blur_radius > 0:
            gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        self.gray_np = np.array(gray, dtype=np.float32) / 255.0
        
        edges_bool = canny(
            image=self.gray_np,
            sigma=canny_sigma,
            low_threshold=canny_low / 255.0,
            high_threshold=canny_high / 255.0
        )
        self.edges = edges_bool.astype(np.uint8)

        self.final_patches = []

    def _compute_metrics(self, bbox):
        l, t, r, b = bbox
        
        area = (r - l) * (b - t)
        
        edge_count = np.sum(self.edges[t:b, l:r])
        
        edge_density = edge_count / max(area, 1)
        
        region_gray = self.gray_np[t:b, l:r]
        variance = np.var(region_gray)
        
        cx = (l + r) / 2
        cy = (t + b) / 2
        z_order = compute_z_order(cx, cy, self.width, self.height)
        
        return edge_density, variance, edge_count, z_order

    def _can_subdivide(self, bbox):
        l, t, r, b = bbox
        w, h = (r - l), (b - t)
        return (w >= 2 * self.min_patch_size) and (h >= 2 * self.min_patch_size)

    def _split_into_quadrants(self, bbox):
        l, t, r, b = bbox
        mx = (l + r) // 2
        my = (t + b) // 2
        return [
            (l,  t,  mx, my),
            (mx, t,  r,  my),
            (l,  my, mx, b),
            (mx, my, r,  b)
        ]

    def build_patches(self, min_variance_threshold=0.001):
        max_uniform_patches = (self.width // self.min_patch_size) * (self.height // self.min_patch_size)
        if self.target_count == max_uniform_patches:
            patches = []
            for y in range(0, self.height, self.min_patch_size):
                for x in range(0, self.width, self.min_patch_size):
                    right = min(x + self.min_patch_size, self.width)
                    bottom = min(y + self.min_patch_size, self.height)
                    
                    bbox = (x, y, right, bottom)
                    density, variance, edge_count, z_order = self._compute_metrics(bbox)
                    
                    patches.append((bbox, density, variance, edge_count, z_order))
            
            self.final_patches = patches
            return self.final_patches
        
        self.final_patches = []
        
        heap = []

        root_bbox = (0, 0, self.width, self.height)
        density, variance, edge_count, z_order = self._compute_metrics(root_bbox)
        
        priority = density if variance >= min_variance_threshold else 0.0
        heapq.heappush(heap, (-priority, root_bbox, density, variance, edge_count, z_order))

        while len(self.final_patches) + len(heap) < self.target_count:
            if not heap:
                break
                
            can_split_any = False
            
            temp_heap = []
            split_candidate = None
            
            while heap and not split_candidate:
                item = heapq.heappop(heap)
                _, bbox, density, variance, edge_count, z_order = item
                
                if self._can_subdivide(bbox):
                    split_candidate = item
                    can_split_any = True
                else:
                    temp_heap.append(item)
            
            for item in temp_heap:
                heapq.heappush(heap, item)
                
            if not can_split_any:
                break
                
            _, bbox, density, variance, edge_count, z_order = split_candidate
            
            quads = self._split_into_quadrants(bbox)
            for sub_box in quads:
                sub_density, sub_variance, sub_edge_count, sub_z = self._compute_metrics(sub_box)
                sub_priority = sub_density if sub_variance >= min_variance_threshold else 0.0
                heapq.heappush(heap, (-sub_priority, sub_box, sub_density, sub_variance, sub_edge_count, sub_z))

        while heap:
            _, bbox, density, variance, edge_count, z_order = heapq.heappop(heap)
            self.final_patches.append((bbox, density, variance, edge_count, z_order))
            
        if len(self.final_patches) > self.target_count:
            self.final_patches.sort(key=lambda x: -x[1])
            self.final_patches = self.final_patches[:self.target_count]
            
        if len(self.final_patches) < self.target_count:
            needed = self.target_count - len(self.final_patches)
            
            grid_size = int(np.ceil(np.sqrt(needed)))
            patch_size = min(self.width, self.height) // grid_size
            patch_size = max(patch_size, self.min_patch_size)
            
            additional_patches = []
            
            existing_areas = [(b[0], b[1], b[2], b[3]) for b, _, _, _, _ in self.final_patches]
            
            for y in range(0, self.height, patch_size):
                if len(additional_patches) >= needed:
                    break
                    
                for x in range(0, self.width, patch_size):
                    right = min(x + patch_size, self.width)
                    bottom = min(y + patch_size, self.height)
                    
                    bbox = (x, y, right, bottom)
                    
                    if right - x < self.min_patch_size or bottom - y < self.min_patch_size:
                        continue
                        
                    overlaps = False
                    for l, t, r, b in existing_areas:
                        if not (x >= r or right <= l or y >= b or bottom <= t):
                            overlaps = True
                            break
                            
                    if not overlaps:
                        density, variance, edge_count, z_order = self._compute_metrics(bbox)
                        additional_patches.append((bbox, density, variance, edge_count, z_order))
                        existing_areas.append(bbox)
                        
                        if len(additional_patches) >= needed:
                            break
            
            self.final_patches.extend(additional_patches)
            
            if len(self.final_patches) < self.target_count:
                still_needed = self.target_count - len(self.final_patches)
                duplicates = random.choices(self.final_patches, k=still_needed)
                self.final_patches.extend(duplicates)
                
        self.final_patches.sort(key=lambda x: x[4])

        return self.final_patches

    def get_leaf_patches(self, patch_count=None, patch_size=16, z_order=True, drop_randomly=True):
        if not self.final_patches:
            self.build_patches()
            
        patch_images = []
        patch_coords = []
        
        for (bbox, *_, z_order_val) in self.final_patches:
            l, t, r, b = bbox
            cropped = self.original_image.crop((int(l), int(t), int(r), int(b)))
            resized = cropped.resize((patch_size, patch_size), Image.LANCZOS)
            patch_images.append(resized)
            
            cx = (l + r) / 2
            cy = (t + b) / 2
            norm_x = 2 * (cx / self.width) - 1
            norm_y = 2 * (cy / self.height) - 1
            patch_coords.append((norm_x, norm_y))

        if patch_count is None:
            return patch_images, patch_coords

        current_count = len(patch_images)
        if current_count > patch_count:
            if drop_randomly:
                indices = np.random.choice(current_count, size=patch_count, replace=False)
                patch_images = [patch_images[i] for i in indices]
                patch_coords = [patch_coords[i] for i in indices]
            else:
                patch_images = patch_images[:patch_count]
                patch_coords = patch_coords[:patch_count]
        elif current_count < patch_count:
            needed = patch_count - current_count
            for _ in range(needed):
                patch_images.append(Image.new("RGB", (patch_size, patch_size), color=(0,0,0)))
                patch_coords.append((0.0, 0.0))

        return patch_images, patch_coords