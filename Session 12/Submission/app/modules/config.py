# Alert: Change these when running in production

# Constants naming convention: All caps separated by underscore
# https://realpython.com/python-constants/

# Where do we store the data?
CHECKPOINT_PATH = "assets/model/model_best_epoch.ckpt"

# Set seed value for reproducibility
SEED = 53

# What is the mean and std deviation of the dataset?
CIFAR_MEAN = (0.4915, 0.4823, 0.4468)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

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
