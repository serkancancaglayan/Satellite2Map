import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4

TRAIN_DATA_DIR = "./maps_dataset/train"
VAL_DATA_DIR = "./maps_dataset/val"
SAVE_EXAMPLE_PATH = "./examples"
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
IMG_SIZE = 256
L1_LAMBDA = 100

NUM_WORKERS = 2
PIN_MEMORY = True

LOAD_CHECKPOINT = True
MODEL_CHECKPOINT_PATH = "model.pt"




