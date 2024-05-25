import torch
import numpy as np
from io import BytesIO
from RBM import RBM
from config import LINK_DATA_HDFS, client, DEVICE, MODEL_SAVE_PATH


def load_data_from_hdfs(hdfs_path, subset_name, batch_size=64):
    """Load data from HDFS."""
    dataset = []
    link_files = f"{hdfs_path}/{subset_name}"
    files = client.list(link_files)
    images = sorted([f for f in files if f.startswith("mnist")])
    labels = sorted([f for f in files if f.startswith("label")])

    for img_file, lbl_file in zip(images, labels):
        link_image = f"{link_files}/{img_file}"
        link_label = f"{link_files}/{lbl_file}"
        # Load image
        with client.read(link_image) as reader:
            image = np.load(BytesIO(reader.read()))
            image = torch.Tensor(image).view(1, 28, 28)

        # Load label
        with client.read(link_label) as reader:
            label = int(reader.read().decode("utf-8"))

        dataset.append((image, label))

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Load data from HDFS
train_loader = load_data_from_hdfs(LINK_DATA_HDFS, "train")
valid_loader = load_data_from_hdfs(LINK_DATA_HDFS, "valid")
test_loader = load_data_from_hdfs(LINK_DATA_HDFS, "test")

# Initialize the RBM model
visible_units = 28 * 28
hidden_units = 64
rbm = RBM(visible_units, hidden_units).to(DEVICE)

# Train the RBM
rbm.fit(train_loader, lr=0.01, epochs=5, k=1)

# Save the trained model
torch.save(rbm.state_dict(), MODEL_SAVE_PATH)
