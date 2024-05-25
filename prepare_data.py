import argparse
import numpy as np
import os
from io import BytesIO
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config.config import (
    LINK_DATA_HDFS,
    LINK_DOWNLOAD_SAVE,
    LINK_DATA_LOCAL,
    NUM_SAMPLES_PER_LABEL,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    CLIENT,
)


def create_dataloader(dataset, batch_size=64):
    """Create a DataLoader for the given dataset."""
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def download_mnist_data(download_path):
    """Download and preprocess MNIST data."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=download_path, train=True, download=True, transform=transform
    )
    return train_dataset


def save_to_hdfs(dataset, hdfs_path):
    CLIENT.makedirs(hdfs_path)
    """Save dataset to HDFS."""
    print("Path hdfs save:", hdfs_path)
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.numpy().flatten()

        # Convert numpy array to bytes
        with BytesIO() as byte_stream:
            np.save(byte_stream, image)
            byte_stream.seek(0)

            # Path to save in HDFS
            hdfs_file_path = f"{hdfs_path}/mnist_{i}.npy"

            # Write to HDFS
            CLIENT.write(hdfs_file_path, data=byte_stream.getvalue(), overwrite=True)

            # Save label
            label_hdfs_path = f"{hdfs_path}/label_{i}.txt"
            CLIENT.write(label_hdfs_path, data=str(label), overwrite=True)


def save_locally(dataset, local_path):
    """Save dataset locally."""
    os.makedirs(local_path, exist_ok=True)  # Create directory if it doesn't exist
    print("Local path:", local_path)
    for i, (image, label) in enumerate(dataset):
        image = image.numpy().flatten()
        np.save(os.path.join(local_path, f"mnist_{i}.npy"), image)
        with open(os.path.join(local_path, f"label_{i}.txt"), "w") as f:
            f.write(str(label))


def cleanup_local_data(local_path):
    """Remove local downloaded data."""
    if os.path.exists(local_path):
        for root, dirs, files in os.walk(local_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(local_path)


def show_dataset_info(dataset):
    """Show information about the dataset."""
    print(f"Number of samples: {len(dataset)}")
    image, label = dataset[0]
    print(f"Sample image shape: {image.shape}")
    print(f"Sample label: {label}")
    print(f"Data type of image: {type(image)}")
    print(f"Data type of label: {type(label)}")


def show_all_labels(dataset):
    """Show all labels in the dataset."""
    labels = [label for _, label in dataset]
    unique_labels = set(labels)
    unique_labels = list(unique_labels)
    print(f"All labels: {unique_labels}")


def split_dataset_by_labels(
    dataset,
    num_samples_per_label=NUM_SAMPLES_PER_LABEL,
    train_ratio=TRAIN_RATIO,
    valid_ratio=VALID_RATIO,
    test_ratio=TEST_RATIO,
):
    """Split dataset by labels."""
    from collections import defaultdict

    label_to_images = defaultdict(list)

    for image, label in dataset:
        if len(label_to_images[label]) < num_samples_per_label:
            label_to_images[label].append((image, label))

    train_set, test_set, valid_set = [], [], []

    for label, images in label_to_images.items():
        num_train = int(len(images) * train_ratio)
        num_test = int(len(images) * valid_ratio)
        num_valid = len(images) - num_train - num_test

        train_set.extend(images[:num_train])
        test_set.extend(images[num_train : num_train + num_test])
        valid_set.extend(images[num_train + num_test :])

    return train_set, valid_set, test_set


def dowload_and_split_data():
    # Download MNIST data
    print("***Download MNIST data***")
    train_dataset = download_mnist_data(LINK_DOWNLOAD_SAVE)
    print("***Dataset info***")
    show_dataset_info(train_dataset)
    show_all_labels(train_dataset)

    # Split dataset
    print("***Split dataset***")
    train_set, valid_set, test_set = split_dataset_by_labels(
        train_dataset, NUM_SAMPLES_PER_LABEL, TRAIN_RATIO, VALID_RATIO, TEST_RATIO
    )
    return train_set, valid_set, test_set


def data_mnist_local():
    train_set, valid_set, test_set = dowload_and_split_data()

    # Save datasets to HDFS
    print("***Save datasets to HDFS***")
    save_locally(train_set, f"{LINK_DATA_LOCAL}/train")
    save_locally(valid_set, f"{LINK_DATA_LOCAL}/valid")
    save_locally(test_set, f"{LINK_DATA_LOCAL}/test")
    print("Success")


def data_mnist_hdfs():
    train_set, valid_set, test_set = dowload_and_split_data()

    # Save datasets to HDFS
    print("***Save datasets to HDFS***")
    save_to_hdfs(train_set, f"{LINK_DATA_HDFS}/train")
    save_to_hdfs(valid_set, f"{LINK_DATA_HDFS}/valid")
    save_to_hdfs(test_set, f"{LINK_DATA_HDFS}/test")
    print("Success")


# Number of samples: 60000
# Sample image shape: torch.Size([1, 28, 28])
# Sample label: 5
# Data type of image: <class 'torch.Tensor'>
# Data type of label: <class 'int'>
# All labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def main():
    parser = argparse.ArgumentParser(description="MNIST Data Preparation")
    parser.add_argument(
        "--save_mode",
        choices=["local", "hdfs"],
        default="local",
        help="Choose where to save the MNIST data: 'local' or 'hdfs'",
    )
    args = parser.parse_args()

    if args.save_mode == "local":
        data_mnist_local()
    elif args.save_mode == "hdfs":
        data_mnist_hdfs()

    # Optional: Cleanup local downloaded data
    # cleanup_local_data(LINK_DOWNLOAD_SAVE)


if __name__ == "__main__":
    main()
