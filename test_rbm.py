def mapper(self, _, file_path):
    # Read image data from HDFS
    with self.client.read(file_path) as reader:
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

