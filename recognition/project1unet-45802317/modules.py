"""
The following code is completing Project 1 outlined in 2025 COMP3710 Pattern
Recognition Report Brief. The main model that will be used to do this will be
an Improved UNet.

The UNet Segmentation Code Demo from Module 3 of the 2025 COMP3710 course
content, by Dr Wei Dai and Dr Shekhar Chandra was observed and adapted as part
of this solution
"""

# Import required modules, mainly related to Pytorch use
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import random

# Import modules required for dataloading and processing the data loaded
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Chose 21 for the random seed because it's my favourite number, no functional
reason
"""
torch.manual_seed(21)
np.random.seed(21)
random.seed(21)
if torch.cuda.is_available():
    torch.cuda.manual_seed(21)

"""
The next step is to create the required visualisation functions. One key
consideration here is that the 2D Oasis data is in grayscale rather than
colour, which changes the normalisation process. To start with, will
use 0.5 as mean value and 0.25 as standard deviation value rather than the
imagenet values (which are for RGB).
"""

# Test tensor to ensure denormalisation is working (will be removed later)
tensor = torch.Tensor([0, 1.5, 0.5, 1])

# Denormalisation function
def denormalisation(tensor):
  mean = torch.tensor([0.5]).view(1,1,1)
  std = torch.tensor([0.25]).view(1,1,1)
  denormalise_tensor = tensor * std + mean
  # Ensure values are between 0 and 1
  return torch.clamp(denormalise_tensor, 0, 1)

# Test output to ensure that denormalisation is working (will be removed later)
print(denormalisation(tensor))
