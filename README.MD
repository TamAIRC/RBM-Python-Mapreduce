# RBM MapReduce in Python with Hadoop

## Project Description

This project demonstrates how to train Restricted Boltzmann Machines (RBMs) on large datasets using the MapReduce programming model with Hadoop 3.3.0. By leveraging the distributed computing capabilities of Hadoop, the project showcases a scalable approach to machine learning tasks that require significant computational resources.

## Features

- Distributed Data Processing: Uses Hadoop MapReduce to partition and process large datasets in parallel.
- RBM Training: Implements training of RBMs on partitioned data chunks using MapReduce.
- Model Aggregation: Aggregates weights from locally trained RBMs to form a global model.
- Scalability: Efficiently handles large-scale data using Hadoop.
- Ease of Use: Provides a straightforward workflow for training and inference with RBMs.

## Requirements

- Python 3.11.x
- Hadoop 3.3.0

## Installation

**1. Clone the repository:**

```bash
git clone https://github.com/TamAIRC/rbm-mapreduce-python.git
cd rbm-mapreduce-python
```

**2. Run setup**

```bash
python setup.py
```

**3. Setup Hadoop 3.3.0:**
Follow the instructions to install and configure Hadoop 3.3.0 on your system. You can find detailed installation steps on the [official Hadoop website]("https://hadoop.apache.org/").

Step by Step setup [there]("link setup")

## Usage

### Downloading and Preparing Data

**1. Run the prepare mnist data script**
[prepare_data.py](./prepare_data.py)
```bash
python prepare_data.py --save_mode local

```

## Project Structure
- `config.py`: Configuration file for paths and HDFS client.
- `setup.py`: 
- `prepare_data.py`: Prepare data script to download, preprocess, and save MNIST data.
- `rbm.py`: 
- `rbm_pretraining.py`: 
- `train.py`: 
<!-- - `mapper.py`: Mapper script for Hadoop MapReduce.
- `reducer.py`: Reducer script for Hadoop MapReduce. -->
- `README.md`: Project description and usage instructions.
- `requirements.txt`: List of Python dependencies.

## Running the Project

To run the project and download, preprocess, and save the MNIST data:

**1. Ensure Hadoop is configured and running:**

```cmd
start-all
```

or

```cmd
start-dfs.sh
start-yarn.sh
```
**2. Run the main script to process and save data locally or to HDFS:**

**RBM Pretrain**
- Use Hadoop: