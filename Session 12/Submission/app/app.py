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

# Import custom modules
import modules.config as config
import numpy as np
import torch
import torchvision
from modules.custom_resnet import CustomResNet
from modules.visualize import convert_back_image, plot_gradcam_images, plot_misclassified_images
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

# Load and initialize the model
model = CustomResNet()
# Using the checkpoint path present in config, load the trained model
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device("cpu")), strict=False)


# Load the misclassified images data
misclassified_image_data = torch.load(config.MISCLASSIFIED_PATH)

inv_normalize = transforms.Normalize(
    mean=[-0.50 / 0.23, -0.50 / 0.23, -0.50 / 0.23], std=[1 / 0.23, 1 / 0.23, 1 / 0.23]
)
classes = config.CIFAR_CLASSES


# def inference(input_img, transparency=0.5, target_layer_number=-1):
#     transform = transforms.ToTensor()
#     org_img = input_img
#     input_img = transform(input_img)
#     input_img = input_img
#     input_img = input_img.unsqueeze(0)
#     outputs = model(input_img)
#     softmax = torch.nn.Softmax(dim=0)
#     o = softmax(outputs.flatten())
#     confidences = {classes[i]: float(o[i]) for i in range(10)}
#     _, prediction = torch.max(outputs, 1)
#     target_layers = [model.layer2[target_layer_number]]
#     cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
#     grayscale_cam = cam(input_tensor=input_img, targets=None)
#     grayscale_cam = grayscale_cam[0, :]
#     img = input_img.squeeze(0)
#     img = inv_normalize(img)
#     rgb_img = np.transpose(img, (1, 2, 0))
#     rgb_img = rgb_img.numpy()
#     visualization = show_cam_on_image(org_img / 255, grayscale_cam, use_rgb=True, image_weight=transparency)
#     return confidences, visualization


def get_target_layer(layer_name):
    """Get target layer for visualization"""
    if layer_name == "prep":
        return [model.prep[-1]]
    elif layer_name == "layer1_x":
        return [model.layer1_x[-1]]
    elif layer_name == "layer1_r1":
        return [model.layer1_r1[-1]]
    elif layer_name == "layer2":
        return [model.layer2[-1]]
    elif layer_name == "layer3_x":
        return [model.layer3_x[-1]]
    elif layer_name == "layer3_r2":
        return [model.layer3_r2[-1]]
    else:
        return None


model_layer_names = ["prep", "layer1_x", "layer1_r1", "layer2", "layer3_x", "layer3_r2"]


def app_interface():
    """Function which provides the Gradio interface"""
    pass


TITLE = "CIFAR10 Image classification using a Custom ResNet Model"
DESCRIPTION = "Gradio App to infer using a Custom ResNet model and get GradCAM results"
examples = [["cat.jpg", 0.5, -1], ["dog.jpg", 0.5, -1]]
demo = gr.Interface(
    app_interface,
    inputs=[
        # This accepts the image after resizing it to 32x32 which is what our model expects
        gr.Image(shape=(32, 32)),
        gr.Number(value=3, maximum=10, minimum=1, step=1.0, precision=0, label="#Classes to show"),
        gr.Checkbox(True, label="Show GradCAM Image"),
        gr.Dropdown(model_layer_names, value="layer3_x", label="Visulalization Layer from Model"),
        # How much should the image be overlayed on the original image
        gr.Slider(0, 1, 0.6, label="Image Overlay Factor"),
        gr.Checkbox(True, label="Show Misclassified Images?"),
        gr.Slider(value=10, maximum=25, minimum=5, step=5.0, precision=0, label="#Misclassified images to show"),
        gr.Checkbox(True, label="Visulize GradCAM for Misclassified images?"),
        gr.Slider(value=10, maximum=25, minimum=5, step=5.0, precision=0, label="#GradCAM images to show"),
    ],
    # outputs=[gr.Label(num_top_classes=3), gr.Image(shape=(32, 32), label="Output").style(width=128, height=128)],
    outputs=[],
    title=TITLE,
    description=DESCRIPTION,
    # examples=examples,
)
demo.launch()
