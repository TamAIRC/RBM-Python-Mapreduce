import os
from hdfs import InsecureClient
import torch

BASE_PATH = os.path.dirname(__file__)
HDFS_URL = "http://localhost:9870/"  # Adjust this to your HDFS namenode URL
HDFS_USER = "TamNgo_2"  # Adjust this to your HDFS user
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL
MODEL_FOLDER = f"{BASE_PATH}\model"
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_SAVE_PATH = f"{MODEL_FOLDER}\\rbm.pth"

# Initialize HDFS client
CLIENT = InsecureClient(HDFS_URL, user=HDFS_USER)

# HDFS paths
LINK_DOWNLOAD_SAVE = "/tmp"
LINK_DATA_HDFS = "/input_mnist"
LINK_MODEL_HDFS = "/model"

# Local paths
LINK_DATA_LOCAL = f"{BASE_PATH}\input_mnist"
LINK_MODEL_LOCAL = f"{BASE_PATH}\model"

# Dataset split ratios
NUM_SAMPLES_PER_LABEL = 100
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

# Ensure output directory exists
CLIENT.makedirs(LINK_DATA_HDFS)
CLIENT.makedirs(LINK_MODEL_HDFS)
