from mrjob.job import MRJob
import numpy as np
from hdfs import InsecureClient
from io import BytesIO


class RBMMapper(MRJob):

    def configure_args(self):
        super(RBMMapper, self).configure_args()
        self.add_passthru_arg(
            "--n_hidden", type=int, default=64, help="Number of hidden units"
        )
        self.add_passthru_arg(
            "--hdfs_path",
            type=str,
            default="hdfs:///input_mnist/train",
            help="HDFS input path",
        )

    def mapper_init(self):
        self.n_visible = 784  # Number of visible units (e.g., for MNIST)
        self.n_hidden = self.options.n_hidden
        self.W = (
            np.random.randn(self.n_visible, self.n_hidden) * 0.01
        )  # Initial weights
        self.learning_rate = 0.01
        self.client = InsecureClient("http://localhost:9870", user="TamNgo_2")

    def mapper(self, _, __):
        hdfs_path = self.options.hdfs_path
        file_list = self.client.list(hdfs_path)

        for file_name in file_list:
            if file_name.endswith(".npy"):
                with self.client.read(f"{hdfs_path}/{file_name}") as reader:
                    byte_stream = BytesIO(reader.read())
                    data = np.load(byte_stream)

                v0 = data  # Visible layer input

                # Positive phase
                h0_prob = self.sigmoid(np.dot(v0, self.W))
                h0_sample = (h0_prob > np.random.rand(self.n_hidden)).astype(np.float32)

                # Negative phase
                v1_prob = self.sigmoid(np.dot(h0_sample, self.W.T))
                h1_prob = self.sigmoid(np.dot(v1_prob, self.W))

                # Compute weight updates
                positive_grad = np.outer(v0, h0_prob)
                negative_grad = np.outer(v1_prob, h1_prob)

                # Emit partial gradients
                grad = positive_grad - negative_grad
                for i in range(self.n_visible):
                    for j in range(self.n_hidden):
                        yield (i, j), grad[i, j]

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == "__main__":
    RBMMapper.run()
