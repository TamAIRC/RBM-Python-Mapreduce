from mrjob.job import MRJob
import numpy as np
from io import BytesIO
import torch
from RBM import RBM
from config import MODEL_SAVE_PATH
import logging


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
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        logging.info("Mapper initialization")

        # Load model
        logging.info("Loading model")
        self.rbm = RBM(visible_units=28 * 28, hidden_units=64)
        self.rbm.load_state_dict(torch.load(self.options.model_path))
        self.rbm.eval()

    def mapper(self, _, line):
        # Process data (assuming line is a serialized image)
        # image = np.load(BytesIO(line))
        logging.info("Processing input data")
        image = np.load(BytesIO(line.encode()))
        image = torch.tensor(image, dtype=torch.float32).view(1, -1)

        # Forward pass
        logging.info("Performing forward pass")
        with torch.no_grad():
            v_prob = self.rbm(image)

        logging.info("Yielding output")
        yield None, v_prob.numpy().tolist()

    def reducer_init(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        logging.info("Reducer initialization")

    def reducer(self, key, values):
        # Aggregate results (example: average output)
        logging.info("Aggregating results")
        aggregated_output = np.mean(list(values), axis=0)
        yield key, aggregated_output


if __name__ == "__main__":
    MRDeepLearningJob.run()
