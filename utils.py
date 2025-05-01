import os
import time
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from scipy import ndimage
from skimage.feature import canny
import heapq
import matplotlib.pyplot as plt

def compute_z_order(x, y, width, height, bits=16):
    nx = float(x) / float(width)
    ny = float(y) / float(height)
    x_bits = int(nx * (2**bits - 1))
    y_bits = int(ny * (2**bits - 1))
    z = 0
    for i in range(bits):
        z |= (x_bits & (1 << i)) << i
        z |= (y_bits & (1 << i)) << (i + 1)
    return z