from mrjob.job import MRJob
import numpy as np
from io import BytesIO
import torch
from RBM import RBM
from config.config import MODEL_SAVE_PATH


class MRDeepLearningJob(MRJob):
    def configure_args(self):
        super(MRDeepLearningJob, self).configure_args()
        self.add_passthru_arg(
            "--model-path",
            type=str,
            default=MODEL_SAVE_PATH,
            help="Path to the model file",
        )

    def mapper(self, _, line):
        # Load model
        rbm = RBM(visible_units=28 * 28, hidden_units=64)
        rbm.load_state_dict(torch.load(self.options.model_path))
        rbm.eval()

        # Process data (assuming line is a serialized image)
        print("Length of line:", len(line))
        image = np.load(BytesIO(line))
        # image = np.load(BytesIO(line.encode()))
        print("Shape of image:", image.shape)
        image = torch.tensor(image, dtype=torch.float32).view(1, -1)

        # Forward pass
        with torch.no_grad():
            v_prob = rbm(image)

        yield None, v_prob.numpy().tolist()

    def reducer(self, key, values):
        # Aggregate results (example: average output)
        aggregated_output = np.mean(list(values), axis=0)
        yield key, aggregated_output


if __name__ == "__main__":
    MRDeepLearningJob.run()
