import os

# Data settings
DATA_DIR = r"C:\Users\Medini\OneDrive\Documents\ml lab project\archive (1)\dataset"  # Full path to your dataset
BATCH_SIZE = 32  # Increased for better stability
NUM_WORKERS = 0  # Set to 0 to avoid shared memory issues
PIN_MEMORY = False  # Set to False for DirectML compatibility

# Model settings
MODEL_NAME = "efficientnet_b2"  # Changed to EfficientNet-B2
PRETRAINED = True
NUM_CLASSES = None  # Will be set automatically based on dataset
FREEZE_BACKBONE = True  # Freeze backbone initially for better transfer learning

# Training settings
EPOCHS = 30  # Increased epochs
LR = 0.001  # Increased learning rate since we're using SGD
WEIGHT_DECAY = 0.0005  # Reduced weight decay
GRADIENT_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 10  # Increased patience
GRADIENT_ACCUMULATION_STEPS = 1  # Removed gradient accumulation for simplicity

# Learning rate scheduler settings
WARMUP_PERCENT = 0.1
DIV_FACTOR = 10
FINAL_DIV_FACTOR = 100

# Data augmentation settings
TRAIN_IMG_SIZE = 260  # Increased for EfficientNet-B2
RANDOM_CROP_SCALE = (0.8, 1.0)  # More aggressive cropping
RANDOM_CROP_RATIO = (0.75, 1.33)  # More varied aspect ratios
ROTATION_DEGREES = 30  # Increased rotation
COLOR_JITTER = (0.2, 0.2, 0.2)  # Increased color jittering
MIXUP_ALPHA = 0.2  # Kept moderate mixup

# Regularization settings
DROPOUT_RATE = 0.2  # Reduced dropout
LABEL_SMOOTHING = 0.1  # Increased label smoothing

# Model save settings
# Define a directory for saving models
MODEL_SAVE_DIR = "saved_models"  # Set a proper directory path

# Update model save paths
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "final_model.pth")

# Ensure the directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Device settings
DEVICE = "dml"  # Can be changed to "cpu" or "cuda" if available

