# Alert: Change these when running in production

# Constants naming convention: All caps separated by underscore
# https://realpython.com/python-constants/

# Where do we store the data?
DATA_PATH = "../../data/"
CHECKPOINT_PATH = "../../checkpoints/"
LOGGING_PATH = "../../logs/"

# Specify the number of epochs
NUM_EPOCHS = 24

# Set the batch size
BATCH_SIZE = 512

# Set seed value for reproducibility
SEED = 53

# What is the start LR and weight decay you'd prefer?
PREFERRED_START_LR = 5e-3
PREFERRED_WEIGHT_DECAY = 1e-5
