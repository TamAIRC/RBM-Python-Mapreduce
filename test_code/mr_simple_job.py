from mrjob.job import MRJob
import numpy as np
import torch
from RBM import RBM
from config.config import MODEL_SAVE_PATH


class MRSimpleJob(MRJob):
    def configure_args(self):
        super(MRSimpleJob, self).configure_args()
        self.add_passthru_arg(
            "--model-path",
            type=str,
            default=MODEL_SAVE_PATH,
            help="Path to the model file",
        )

    def mapper_init(self):
        self.rbm = RBM(visible_units=28 * 28, hidden_units=64)
        self.rbm.load_state_dict(torch.load(self.options.model_path))
        self.rbm.eval()

    def mapper(self, _, line):
        yield "line_length", len(line)

    def reducer(self, key, values):
        yield key, sum(values)


if __name__ == "__main__":
    MRSimpleJob.run()

