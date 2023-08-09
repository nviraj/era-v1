---
title: Era Week12 Lightning.resnet
emoji: 🐠
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 3.39.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# CIFAR10 Image classification using a Custom ResNet Model

## What is the app about?

[This app](https://huggingface.co/spaces/nviraj/ERA-V1-Assignment12) built using [Gradio](https://www.gradio.app/) provides an interface to run inferences for CIFAR10 image classification using a custom ResNet model trained using PyTorch and Lightning with \>90% accuracy.

### What input does it require?

- **Example Input**
  - Please note that example inputs have been provided for you to test the app below the Submit button. Please select one of the examples and hit Submit to see the app in action
- **Inference Related**
  - Image
    - Any image for the following 10 CIFAR10 classes [airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks]
    - It accepts any resolution and image type
  - How many top classes to predict?
    - Max of 10 classes
  - Do you want to show the GradCAM image?
    - This shows you features deemed important by the model in making the prediction
  - Which layer of the model do you want to generate GradCAM for?
    - A network has multiple layers and it is sequentially shown as a drop down. Every layer incrementally identifies bigger parts of the image. Have fun generating the visualization for different layers.
  - By what factor do you want to overlay the original image on GradCAM?
    - Smaller the factor more prominent is GradCAM Hotspot.
    - As you increase the factor the original image becomes more opaque and prominent over the heatmap
- **Diagnostics Related**
  - Do you want to show Misclassified Images and how many?
    - This comes in handy to see where the model fails to predict accurate classes
  - Do you want to see GradCAM for Misclassified Images and how many?
    - This is useful to see what parts of the image led to incorrect classification

### What is the output?

- Predictions for top number of classes chosen as well as the predicted class
- Either the original image or image + GradCAM heatmap based on input chosen
- Misclassified Images by the model
- GradCAM for Misclassified Images by the model

### How was the model built?

- Model was trained using a custom ResNet model trained for just 24 epochs with 91.4% validation accuracy
- The code can be found here
  - [Notebook](https://github.com/nviraj/era-v1/blob/main/Session%2012/Submission/ERA%20V1%20-%20Viraj%20-%20Assignment%2012.ipynb)
  - [Modules](https://github.com/nviraj/era-v1/tree/main/Session%2012/Submission/modules)
  - [Model](https://github.com/nviraj/era-v1/tree/main/Session%2012/Submission/models)

### Links

- [GradCAM?](https://arxiv.org/abs/1610.02391)
- [Pytorch](https://pytorch.org/)
- [Pytorch Lightning](https://www.pytorchlightning.ai/index.html)
- [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
