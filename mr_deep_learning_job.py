from mrjob.job import MRJob
import numpy as np
from io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
from RBM import RBM
from config import MODEL_SAVE_PATH


class DNN(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class MRDeepLearningJob(MRJob):
    def configure_args(self):
        super(MRDeepLearningJob, self).configure_args()
        self.add_passthru_arg(
            "--model-path",
            type=str,
            default=MODEL_SAVE_PATH,
            help="Path to the model file",
        )

    def mapper_init(self):
        # Load the pre-trained RBM model
        self.rbm = RBM(visible_units=28 * 28, hidden_units=64)
        self.rbm.load_state_dict(torch.load(self.options.model_path))
        self.rbm.eval()

        # Initialize the DNN with pre-trained weights from RBM
        self.dnn = DNN(
            input_size=28 * 28, hidden_units=64, output_size=10
        )  # Assuming 10 classes for classification
        self.dnn.fc1.weight.data = self.rbm.W.data
        self.dnn.fc1.bias.data = self.rbm.h_bias.data

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.dnn.parameters(), lr=0.01, momentum=0.9)

    def mapper(self, _, line):
        # Process data (assuming line is a serialized image and label)
        image, label = line.split(",")
        image = np.load(BytesIO(image.encode("latin1")))
        label = int(label)
        image = torch.tensor(image, dtype=torch.float32).view(1, -1)
        label = torch.tensor([label], dtype=torch.long)

        # Forward pass
        output = self.dnn(image)
        loss = self.criterion(output, label)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        yield None, (
            self.dnn.fc1.weight.data.numpy().tolist(),
            self.dnn.fc2.weight.data.numpy().tolist(),
        )

    def reducer(self, key, values):
        # Aggregate results by averaging the weights
        fc1_weights = []
        fc2_weights = []
        for v in values:
            fc1_weights.append(np.array(v[0]))
            fc2_weights.append(np.array(v[1]))

        avg_fc1_weights = np.mean(fc1_weights, axis=0).tolist()
        avg_fc2_weights = np.mean(fc2_weights, axis=0).tolist()

        yield "fc1_weights", avg_fc1_weights
        yield "fc2_weights", avg_fc2_weights


if __name__ == "__main__":
    MRDeepLearningJob.run()
# python mr_deep_learning_job.py -r hadoop hdfs:///input_mnist/train --model-path hdfs:///model/rbm.pth --hadoop-streaming-jar /path/to/hadoop-streaming.jar
