import torch
import numpy as np
from io import BytesIO
from RBM import RBM
from config.config import (
    LINK_DATA_HDFS,
    CLIENT,
    DEVICE,
    LINK_DOWNLOAD_SAVE,
    MODEL_SAVE_PATH,
)
from prepare_data import create_dataloader, download_mnist_data, split_dataset_by_labels


def load_data_from_hdfs(hdfs_path, subset_name, batch_size=64):
    """Load data from HDFS."""
    dataset = []
    link_files = f"{hdfs_path}/{subset_name}"
    files = CLIENT.list(link_files)
    images = sorted([f for f in files if f.startswith("mnist")])
    labels = sorted([f for f in files if f.startswith("label")])

    for img_file, lbl_file in zip(images, labels):
        link_image = f"{link_files}/{img_file}"
        link_label = f"{link_files}/{lbl_file}"
        # Load image
        with CLIENT.read(link_image) as reader:
            image = np.load(BytesIO(reader.read()))
            image = torch.Tensor(image).view(1, 28, 28)

        # Load label
        with CLIENT.read(link_label) as reader:
            label = int(reader.read().decode("utf-8"))

        dataset.append((image, label))

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    # Load data from HDFS
    print("***Load data***")
    # train_loader = load_data_from_hdfs(LINK_DATA_HDFS, "train")

    # Download MNIST data
    train_dataset = download_mnist_data(LINK_DOWNLOAD_SAVE)

    # Split dataset
    train_set, valid_set, test_set = split_dataset_by_labels(train_dataset)
    train_loader = create_dataloader(train_set)
    # Initialize the RBM model
    visible_units = 28 * 28
    hidden_units = 64
    rbm = RBM(visible_units, hidden_units).to(DEVICE)

    # Train the RBM
    print("***Training model***")
    rbm.fit(train_loader, lr=0.01, epochs=50, k=1)

    # Save the trained model
    print("***Save model***")
    torch.save(rbm.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
