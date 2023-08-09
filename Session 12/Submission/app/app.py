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

# import torchvision
from modules.custom_resnet import CustomResNet
from modules.visualize import plot_gradcam_images, plot_misclassified_images
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

# Load and initialize the model
model = CustomResNet()

# Define device
cpu = torch.device("cpu")

# Using the checkpoint path present in config, load the trained model
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=cpu), strict=False)
# Send model to CPU
model.to(cpu)
# Make the model in evaluation mode
model.eval()
# print(f"Model Device: {next(model.parameters()).device}")


# Load the misclassified images data
misclassified_image_data = torch.load(config.MISCLASSIFIED_PATH, map_location=cpu)

# Class Names
classes = list(config.CIFAR_CLASSES)
# Allowed model names
model_layer_names = ["prep", "layer1_x", "layer1_r1", "layer2", "layer3_x", "layer3_r2"]


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


def generate_prediction(input_image, num_classes=3, show_gradcam=True, transparency=0.6, layer_name="layer3_x"):
    """ "Given an input image, generate the prediction, confidence and display_image"""
    mean = list(config.CIFAR_MEAN)
    std = list(config.CIFAR_STD)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    with torch.no_grad():
        orginal_img = input_image
        input_image = transform(input_image).unsqueeze(0).to(cpu)
        # print(f"Input Device: {input_image.device}")
        model_output = model(input_image).to(cpu)
        # print(f"Output Device: {outputs.device}")
        output_exp = torch.exp(model_output).to(cpu)
        # print(f"Output Exp Device: {o.device}")

        output_numpy = np.squeeze(np.asarray(output_exp.numpy()))
        # get indexes of probabilties in descending order
        sorted_indexes = np.argsort(output_numpy)[::-1]
        # sort the probabilities in descending order
        # final_class = classes[o_np.argmax()]

        confidences = {}
        for _ in range(int(num_classes)):
            # set the confidence of highest class with highest probability
            confidences[classes[sorted_indexes[_]]] = float(output_numpy[sorted_indexes[_]])

    # Show Grad Cam
    if show_gradcam:
        # Get the target layer
        target_layers = get_target_layer(layer_name)
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        cam_generated = cam(input_tensor=input_image, targets=None)
        cam_generated = cam_generated[0, :]
        display_image = show_cam_on_image(orginal_img / 255, cam_generated, use_rgb=True, image_weight=transparency)
    else:
        display_image = orginal_img

    return confidences, display_image


def app_interface(
    input_image,
    num_classes,
    show_gradcam,
    layer_name,
    transparency,
    show_misclassified,
    num_misclassified,
    show_gradcam_misclassified,
    num_gradcam_misclassified,
):
    """Function which provides the Gradio interface"""

    # Get the prediction for the input image along with confidence and display_image
    confidences, display_image = generate_prediction(input_image, num_classes, show_gradcam, transparency, layer_name)

    if show_misclassified:
        misclassified_fig, misclassified_axs = plot_misclassified_images(
            data=misclassified_image_data, class_label=classes, num_images=num_misclassified
        )
    else:
        misclassified_fig = None

    if show_gradcam_misclassified:
        gradcam_fig, gradcam_axs = plot_gradcam_images(
            model=model,
            data=misclassified_image_data,
            class_label=classes,
            # Use penultimate block of resnet18 layer 3 as the target layer for gradcam
            # Decided using model summary so that dimensions > 7x7
            target_layers=get_target_layer(layer_name),
            targets=None,
            num_images=num_gradcam_misclassified,
            image_weight=transparency,
        )
    else:
        gradcam_fig = None

    # # delete ununsed axises
    # del misclassified_axs
    # del gradcam_axs

    return confidences, display_image, misclassified_fig, gradcam_fig


TITLE = "CIFAR10 Image classification using a Custom ResNet Model"
DESCRIPTION = "Gradio App to infer using a Custom ResNet model and get GradCAM results"
examples = [
    ["assets/images/airplane.jpg", 3, True, "layer3_x", 0.6, True, 5, True, 5],
    ["assets/images/bird.jpeg", 4, True, "layer3_x", 0.7, True, 10, True, 20],
    ["assets/images/car.jpg", 5, True, "layer3_x", 0.5, True, 15, True, 5],
    ["assets/images/cat.jpeg", 6, True, "layer3_x", 0.65, True, 20, True, 10],
    ["assets/images/deer.jpg", 7, False, "layer2", 0.75, True, 5, True, 5],
    ["assets/images/dog.jpg", 8, True, "layer2", 0.55, True, 10, True, 5],
    ["assets/images/frog.jpeg", 9, True, "layer2", 0.8, True, 15, True, 15],
    ["assets/images/horse.jpg", 10, False, "layer1_r1", 0.85, True, 20, True, 5],
    ["assets/images/ship.jpg", 3, True, "layer1_r1", 0.4, True, 5, True, 15],
    ["assets/images/truck.jpg", 4, True, "layer1_r1", 0.3, True, 5, True, 10],
]
inference_app = gr.Interface(
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
    outputs=[
        gr.Label(label="Confidences", container=True, show_label=True),
        gr.Image(shape=(32, 32), label="Grad CAM/ Input Image", container=True, show_label=True).style(
            width=256, height=256
        ),
        gr.Plot(label="Misclassified images", container=True, show_label=True),
        gr.Plot(label="Grad CAM of Misclassified images", container=True, show_label=True),
    ],
    title=TITLE,
    description=DESCRIPTION,
    examples=examples,
)
inference_app.launch()
