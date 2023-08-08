# Outline
# Import packages
# Import modules
# Constants
# Load model
# Function to process user uploaded image/ examples
# Inference function
# Gradio examples
# Gradio App

# Import packages required for the app
import gradio as gr
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from resnet import ResNet18
from torchvision import transforms
