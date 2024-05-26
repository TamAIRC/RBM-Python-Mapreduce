from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np
from hdfs import InsecureClient
from io import BytesIO


class RBMJob(MRJob):

    def configure_args(self):
        super(RBMJob, self).configure_args()
        self.add_passthru_arg(
            "--n_hidden", type=int, default=64, help="Number of hidden units"
        )
        self.add_passthru_arg(
            "--hdfs_path",
            type=str,
            default="hdfs:///input_mnist/train",
            help="HDFS input path",
        )
        self.add_passthru_arg(
            "--hdfs_output_path",
            type=str,
            default="hdfs:///output_rbm_weights",
            help="HDFS output path",
        )

    def mapper_init(self):
        self.n_visible = 784  # Number of visible units (e.g., for MNIST)
        self.n_hidden = self.options.n_hidden
        self.W = (
            np.random.randn(self.n_visible, self.n_hidden) * 0.01
        )  # Initial weights
        self.learning_rate = 0.01
        self.client = InsecureClient("http://localhost:9870", user="TamNgo_2")
        self.hdfs_path = self.options.hdfs_path
        self.file_list = self.client.list(self.hdfs_path)

    def mapper(self, _, __):
        for file_name in self.file_list:
            if file_name.endswith(".npy"):
                with self.client.read(f"{self.hdfs_path}/{file_name}") as reader:
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

    def reducer_init(self):
        self.n_visible = 784  # Number of visible units (e.g., for MNIST)
        self.n_hidden = self.options.n_hidden
        self.W = np.zeros((self.n_visible, self.n_hidden))
        self.client = InsecureClient("http://localhost:9870", user="TamNgo_2")
        self.hdfs_output_path = self.options.hdfs_output_path

    def reducer(self, key, values):
        i, j = key
        self.W[i, j] += sum(values)  # Sum up partial gradients
    def reducer_final(self):
        # Save the updated weights to HDFS
        with BytesIO() as byte_stream:
            np.save(byte_stream, self.W)
            byte_stream.seek(0)
            # f"{self.hdfs_output_path}/rbm_weights.pth"
            self.client.write(
                f"/rbm_weights.pth",
                data=byte_stream.getvalue(),
                overwrite=True,
            )

    def steps(self):
        return [
            MRStep(
                mapper_init=self.mapper_init,
                mapper=self.mapper,
                reducer_init=self.reducer_init,
                reducer=self.reducer,
                reducer_final=self.reducer_final,
            )
        ]

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == "__main__":
    RBMJob.run()
# python RBMJob.py --n_hidden 64 --hdfs_path /input_mnist/train --hdfs_output_path hdfs:///output_rbm_weights mnist_198.npy mnist_199.npy