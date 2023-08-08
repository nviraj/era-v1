# Alert: Change these when running in production

# Constants naming convention: All caps separated by underscore
# https://realpython.com/python-constants/

# Where do we store the data?
DATA_PATH = "../../data/"
CHECKPOINT_PATH = "../../checkpoints/"
LOGGING_PATH = "../../logs/"
MISCLASSIFIED_PATH = "Misclassified_Data.pt"
MODEL_PATH = "CustomResNet.pt"

# Specify the number of epochs
NUM_EPOCHS = 24

# Set the batch size
BATCH_SIZE = 512

# Set seed value for reproducibility
SEED = 53

# What is the start LR and weight decay you'd prefer?
PREFERRED_START_LR = 5e-3
PREFERRED_WEIGHT_DECAY = 1e-5


# What is the mean and std deviation of the dataset?
CIFAR_MEAN = (0.4915, 0.4823, 0.4468)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# What is the cutout size?
CUTOUT_SIZE = 16

# What are the classes in CIFAR10?
# Create class labels and convert to tuple
CIFAR_CLASSES = tuple(
    c.capitalize()
    for c in [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
)
