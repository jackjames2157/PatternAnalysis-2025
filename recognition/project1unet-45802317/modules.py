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

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Chose 21 for the random seed because it's my favourite number, no functional reason
"""
torch.manual_seed(21)
np.random.seed(21)
random.seed(21)
if torch.cuda.is_available():
    torch.cuda.manual_seed(21)
