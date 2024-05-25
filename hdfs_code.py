from config import client


def remove_hdfs_files(hdfs_path):
    """Remove all MNIST files from HDFS."""
    try:
        # List all files in the directory
        files = client.list(hdfs_path)

        # Filter out the MNIST files
        mnist_files = [f for f in files if f.startswith("mnist_")]

        # Remove each file
        for file in mnist_files:
            client.delete(os.path.join(hdfs_path, file), recursive=False)
        print(f"Removed {len(mnist_files)} files from {hdfs_path}")
    except Exception as e:
        print(f"Error removing files: {e}")


def main():
    # Cleanup local downloaded data
    remove_hdfs_files("/input*")


if __name__ == "__main__":
    main()
