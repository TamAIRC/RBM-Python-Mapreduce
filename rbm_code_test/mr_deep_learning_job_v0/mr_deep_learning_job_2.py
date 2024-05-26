from mrjob.job import MRJob
import numpy as np
from io import BytesIO
import torch
from RBM import RBM
# from config import MODEL_SAVE_PATH
MODEL_SAVE_PATH = "D:/01_thacsi/src/BigData/hadoop_rbm/model/rbm_fine_tuned.pkl"
class MRDeepLearningJob(MRJob):

    def configure_args(self):
        super(MRDeepLearningJob, self).configure_args()
        self.add_passthru_arg('--model-path', type=str, default=MODEL_SAVE_PATH, help='Path to the model file')

    def mapper_init(self):
        # Load the RBM model
        self.rbm = RBM(visible_units=28 * 28, hidden_units=64)
        self.rbm.load_state_dict(torch.load(self.options.model_path))
        self.rbm.eval()

    def mapper(self, _, line):
        # Assuming each line is a serialized numpy array of the image
        image = np.load(BytesIO(line.encode('latin1')))
        image = torch.tensor(image, dtype=torch.float32).view(1, -1)

        # Forward pass
        with torch.no_grad():
            v_prob = self.rbm.v_given_h(image)

        yield None, v_prob.numpy().tolist()

    def reducer(self, key, values):
        # Aggregate results by averaging the outputs
        aggregated_output = np.mean(list(values), axis=0)
        yield key, aggregated_output.tolist()

if __name__ == '__main__':
    MRDeepLearningJob.run()
