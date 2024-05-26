from io import BytesIO
from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np
from hdfs import InsecureClient

from config.config import CLIENT

class RBMReducer(MRJob):

    def configure_args(self):
        super(RBMReducer, self).configure_args()
        self.add_passthru_arg(
            "--n_hidden", type=int, default=64, help="Number of hidden units"
        )
        self.add_passthru_arg(
            "--n_visible", type=int, default=784, help="Number of visible units"
        )
        self.add_passthru_arg(
            "--hdfs_output_path", type=str, default="hdfs:///output_rbm_weights", help="HDFS output path"
        )

    def reducer_init(self):
        self.n_visible = self.options.n_visible
        self.n_hidden = self.options.n_hidden
        self.W = np.zeros((self.n_visible, self.n_hidden))
        self.client = CLIENT

    def reducer(self, key, values):
        i, j = key
        self.W[i, j] += sum(values)  # Sum up partial gradients

    def reducer_final(self):
        # Save the updated weights to HDFS
        hdfs_output_path = self.options.hdfs_output_path
        with BytesIO() as byte_stream:
            np.save(byte_stream, self.W)
            byte_stream.seek(0)
            self.client.write(f"{hdfs_output_path}/rbm_weights.npy", data=byte_stream.getvalue(), overwrite=True)

    def steps(self):
        return [
            MRStep(
                reducer_init=self.reducer_init,
                reducer=self.reducer,
                reducer_final=self.reducer_final,
            )
        ]

if __name__ == "__main__":
    RBMReducer.run()
