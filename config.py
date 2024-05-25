import os
from hdfs import InsecureClient
import torch

BASE_PATH = os.path.dirname(__file__)
HDFS_URL = "http://localhost:9870/"  # Adjust this to your HDFS namenode URL
# HDFS_USER = "hdfs"  # Adjust this to your HDFS user
HDFS_USER = "TamNgo_2"  # Adjust this to your HDFS user
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL
MODEL_FOLDER = f"{BASE_PATH}\model"
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_SAVE_PATH = f"{MODEL_FOLDER}\\rbm.pth"

# Initialize HDFS client
client = InsecureClient(HDFS_URL, user=HDFS_USER)

# HDFS paths
LINK_DOWNLOAD_SAVE = "/tmp"
LINK_DATA_HDFS = "/input_mnist"
LINK_MODEL_HDFS = "/model"

LINK_DATA_LOCAL = f"{BASE_PATH}\input_mnist"
LINK_MODEL_LOCAL = f"{BASE_PATH}\model"

INPUT_DATA = "/input_btl/mnist_train_500.json"  # Path to the input data in HDFS

# Ensure output directory exists
client.makedirs(LINK_DATA_HDFS)
client.makedirs(LINK_MODEL_HDFS)
